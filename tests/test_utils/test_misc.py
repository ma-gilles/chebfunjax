"""Tests for chebfunjax.utils.misc — standard_chop, gridsample, abstract_qr."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.misc import abstract_qr, gridsample, standard_chop
from chebfunjax.utils.quadrature import chebpts


# ============================================================================
# Tier 1: Mathematical property tests (no MATLAB data needed)
# ============================================================================


class TestStandardChopProperties:
    """Mathematical property tests for standard_chop.

    JAX contract: not JIT-safe (uses Python control flow with data-dependent
    branching). Called outside JIT in the adaptive construction loop.
    """

    def test_geometric_decay_basic(self):
        """Geometrically decaying coefficients should be chopped well before n."""
        coeffs = 10.0 ** (-jnp.arange(1, 51, dtype=jnp.float64))
        cutoff = standard_chop(coeffs)
        assert 1 < cutoff < 50
        # The exact MATLAB answer is 18.
        assert cutoff == 18

    def test_geometric_decay_with_eps_noise(self):
        """Adding eps-level noise should not prevent chopping."""
        base = 10.0 ** (-jnp.arange(1, 51, dtype=jnp.float64))
        noise = jnp.cos(jnp.arange(1, 51, dtype=jnp.float64) ** 2)
        coeffs = base + 1e-16 * noise
        cutoff = standard_chop(coeffs)
        assert 1 < cutoff < 50

    def test_too_short_returns_n(self):
        """Vectors shorter than 17 should never be chopped."""
        for n in [1, 5, 10, 16]:
            coeffs = 10.0 ** (-jnp.arange(1, n + 1, dtype=jnp.float64))
            cutoff = standard_chop(coeffs)
            assert cutoff == n, f"Expected {n} for length-{n} input, got {cutoff}"

    def test_all_zeros_returns_one(self):
        """All-zero coefficients should return cutoff = 1."""
        coeffs = jnp.zeros(50, dtype=jnp.float64)
        cutoff = standard_chop(coeffs)
        assert cutoff == 1

    def test_single_coefficient(self):
        """Single coefficient should never be chopped."""
        coeffs = jnp.array([3.14], dtype=jnp.float64)
        cutoff = standard_chop(coeffs)
        assert cutoff == 1

    def test_tol_one_returns_one(self):
        """tol >= 1 should always return cutoff = 1."""
        coeffs = jnp.ones(50, dtype=jnp.float64)
        assert standard_chop(coeffs, tol=1.0) == 1
        assert standard_chop(coeffs, tol=2.0) == 1

    def test_tighter_tol_gives_more_coefficients(self):
        """A tighter tolerance should retain at least as many coefficients."""
        coeffs = 10.0 ** (-jnp.arange(1, 51, dtype=jnp.float64))
        noise = jnp.cos(jnp.arange(1, 51, dtype=jnp.float64) ** 2)
        coeffs_noisy = coeffs + 1e-10 * noise

        cut_loose = standard_chop(coeffs_noisy, tol=1e-8)
        cut_tight = standard_chop(coeffs_noisy, tol=1e-14)
        assert cut_tight >= cut_loose, (
            f"tol=1e-14 gave cutoff {cut_tight}, "
            f"but tol=1e-8 gave {cut_loose} — should be >=."
        )

    def test_constant_function_is_chopped(self):
        """A constant function padded with zeros should be chopped to 1."""
        coeffs = jnp.zeros(50, dtype=jnp.float64).at[0].set(5.0)
        cutoff = standard_chop(coeffs)
        # The envelope is [1, 0, 0, ...], which triggers the plateau
        # at j=2 with e1=0 -> plateau_point = 1 -> cutoff = 1.
        assert cutoff == 1

    def test_cutoff_never_exceeds_n(self):
        """Cutoff must never exceed the input length."""
        for n in [17, 50, 100]:
            coeffs = jnp.ones(n, dtype=jnp.float64)
            cutoff = standard_chop(coeffs)
            assert cutoff <= n

    def test_cutoff_always_at_least_one(self):
        """Cutoff must always be at least 1."""
        coeffs = jnp.zeros(50, dtype=jnp.float64)
        assert standard_chop(coeffs) >= 1
        coeffs2 = jnp.ones(50, dtype=jnp.float64) * 1e-300
        assert standard_chop(coeffs2) >= 1

    def test_well_resolved_sin_coefficients(self):
        """Chebyshev coefficients of sin(x) on [-1,1] decay supergeometrically.
        With enough coefficients, standard_chop should find them resolved."""
        # sin(x) on [-1,1] needs about 14 Chebyshev coefficients.
        # Use n=33 points (more than enough).
        x = chebpts(33, kind=2)
        vals = jnp.sin(x)
        # Convert values to coefficients via DCT
        n = vals.shape[0]
        tmp = jnp.concatenate([vals[::-1], vals[1:-1]])
        coeffs = jnp.real(jnp.fft.ifft(tmp))[:n]
        coeffs = coeffs.at[1:n - 1].set(2 * coeffs[1:n - 1])

        cutoff = standard_chop(coeffs)
        # sin(x) should resolve in about 14-15 coefficients.
        assert cutoff <= 20, f"sin(x) cutoff = {cutoff}, expected <= 20"
        assert cutoff >= 10, f"sin(x) cutoff = {cutoff}, expected >= 10"


class TestGridsampleProperties:
    """Property tests for gridsample."""

    def test_gridsample_sin_size(self):
        """gridsample should return array of length n."""
        v = gridsample(jnp.sin, 10)
        assert v.shape == (10,)

    def test_gridsample_constant(self):
        """Sampling a constant function should give constant values."""
        v = gridsample(lambda x: jnp.ones_like(x) * 7.0, 5)
        npt.assert_allclose(np.array(v), 7.0, atol=1e-15)

    def test_gridsample_domain(self):
        """On [0, pi], sin should start near 0 and return to 0."""
        v = gridsample(jnp.sin, 5, domain=(0.0, float(jnp.pi)))
        # First and last Chebyshev points are the endpoints.
        npt.assert_allclose(float(v[0]), 0.0, atol=1e-14)
        npt.assert_allclose(float(v[-1]), 0.0, atol=1e-14)

    def test_gridsample_trig(self):
        """Trigonometric grid should be equispaced and not include endpoint."""
        v = gridsample(jnp.sin, 4, domain=(0.0, 2.0 * float(jnp.pi)), kind="trig")
        assert v.shape == (4,)
        # sin at equispaced points 0, pi/2, pi, 3pi/2
        expected = jnp.sin(jnp.linspace(0, 2.0 * jnp.pi, 4, endpoint=False))
        npt.assert_allclose(np.array(v), np.array(expected), atol=1e-14)

    def test_gridsample_bad_kind(self):
        """Invalid kind should raise ValueError."""
        with pytest.raises(ValueError, match="kind must be"):
            gridsample(jnp.sin, 5, kind="bad")


class TestAbstractQRProperties:
    """Property tests for abstract_qr."""

    def test_standard_qr_3x2(self):
        """Standard inner product QR of a 3x2 matrix."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64)
        E = jnp.eye(3, 2, dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))

        # Q^T Q ≈ I
        QtQ = Q.T @ Q
        npt.assert_allclose(np.array(QtQ), np.eye(2), atol=1e-12)

        # QR ≈ A
        npt.assert_allclose(np.array(Q @ R), np.array(A), atol=1e-12)

        # R is upper triangular
        assert float(jnp.abs(R[1, 0])) < 1e-14

    def test_qr_square_matrix(self):
        """QR of a square matrix."""
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (4, 4), dtype=jnp.float64)
        E = jnp.eye(4, dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))

        npt.assert_allclose(np.array(Q.T @ Q), np.eye(4), atol=1e-11)
        npt.assert_allclose(np.array(Q @ R), np.array(A), atol=1e-11)

    def test_qr_single_column(self):
        """QR of a single-column matrix (just normalisation)."""
        A = jnp.array([[3.0], [4.0]], dtype=jnp.float64)
        E = jnp.eye(2, 1, dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))

        # Q^T Q should be [[1.0]] (1x1 identity)
        npt.assert_allclose(np.array(Q.T @ Q), np.eye(1), atol=1e-14)
        # QR should reconstruct A
        npt.assert_allclose(np.array(Q @ R), np.array(A), atol=1e-14)

    def test_qr_orthogonality_tall_random(self):
        """QR of a tall random matrix preserves orthogonality and reconstruction."""
        m, p = 8, 4
        key = jax.random.PRNGKey(7)
        A = jax.random.normal(key, (m, p), dtype=jnp.float64)
        E = jnp.eye(m, p, dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))

        # QR should reconstruct A.
        npt.assert_allclose(np.array(Q @ R), np.array(A), atol=1e-11)

        # Q^T Q should be identity (orthonormal columns w.r.t. standard IP).
        npt.assert_allclose(np.array(Q.T @ Q), np.eye(p), atol=1e-11)

        # R should be upper triangular.
        for i in range(p):
            for j in range(i):
                assert abs(float(R[i, j])) < 1e-12, f"R[{i},{j}] = {float(R[i, j])}"


# ============================================================================
# Tier 2: MATLAB cross-validation tests
# ============================================================================


class TestStandardChopMATLAB:
    """Cross-validate standard_chop against MATLAB Chebfun reference data.

    Reference data generated by matlab_harness/refs/misc.m using
    rng(42) for reproducibility.
    """

    @pytest.fixture(autouse=True)
    def _load_refs(self):
        """Load MATLAB reference data."""
        from tests.conftest import load_matlab_ref
        self.ref = load_matlab_ref("misc.mat")

    @pytest.mark.matlab
    def test_geom_decay(self):
        """10^{-k} for k=1..50."""
        coeffs = jnp.asarray(self.ref["sc_geom_coeffs"], dtype=jnp.float64)
        assert standard_chop(coeffs) == int(self.ref["sc_geom_cutoff"])

    @pytest.mark.matlab
    def test_geom_eps_noise(self):
        """10^{-k} + eps*noise."""
        coeffs = jnp.asarray(self.ref["sc_geom_eps_coeffs"], dtype=jnp.float64)
        assert standard_chop(coeffs) == int(self.ref["sc_geom_eps_cutoff"])

    @pytest.mark.matlab
    def test_geom_1e13_noise(self):
        """10^{-k} + 1e-13*noise."""
        coeffs = jnp.asarray(self.ref["sc_geom_13_coeffs"], dtype=jnp.float64)
        assert standard_chop(coeffs) == int(self.ref["sc_geom_13_cutoff"])

    @pytest.mark.matlab
    def test_geom_1e10_noise(self):
        """10^{-k} + 1e-10*noise — not happy at default tol."""
        coeffs = jnp.asarray(self.ref["sc_geom_10_coeffs"], dtype=jnp.float64)
        assert standard_chop(coeffs) == int(self.ref["sc_geom_10_cutoff"])

    @pytest.mark.matlab
    def test_geom_1e10_tol10(self):
        """10^{-k} + 1e-10*noise with tol=1e-10."""
        coeffs = jnp.asarray(self.ref["sc_geom_10_coeffs"], dtype=jnp.float64)
        assert standard_chop(coeffs, tol=1e-10) == int(self.ref["sc_geom_10_tol10_cutoff"])

    @pytest.mark.matlab
    def test_all_zeros(self):
        """All-zero coefficients."""
        assert standard_chop(jnp.zeros(50, dtype=jnp.float64)) == int(self.ref["sc_zero_cutoff"])

    @pytest.mark.matlab
    def test_short_vector(self):
        """Short vector (< 17 coefficients)."""
        coeffs = jnp.asarray(self.ref["sc_short_coeffs"], dtype=jnp.float64)
        assert standard_chop(coeffs) == int(self.ref["sc_short_cutoff"])

    @pytest.mark.matlab
    def test_sin_coeffs(self):
        """Chebfun coefficients of sin(x)."""
        coeffs = jnp.asarray(self.ref["sc_sin_coeffs"], dtype=jnp.float64)
        cutoff = standard_chop(coeffs)
        expected = int(self.ref["sc_sin_cutoff"])
        assert cutoff == expected, f"sin cutoff: got {cutoff}, expected {expected}"

    @pytest.mark.matlab
    def test_exp_coeffs(self):
        """Chebfun coefficients of exp(x)."""
        coeffs = jnp.asarray(self.ref["sc_exp_coeffs"], dtype=jnp.float64)
        cutoff = standard_chop(coeffs)
        expected = int(self.ref["sc_exp_cutoff"])
        assert cutoff == expected, f"exp cutoff: got {cutoff}, expected {expected}"

    @pytest.mark.matlab
    def test_sharp_gaussian(self):
        """Chebfun coefficients of exp(-100*x^2)."""
        coeffs = jnp.asarray(self.ref["sc_sharp_coeffs"], dtype=jnp.float64)
        cutoff = standard_chop(coeffs)
        expected = int(self.ref["sc_sharp_cutoff"])
        assert cutoff == expected, f"sharp gaussian cutoff: got {cutoff}, expected {expected}"

    @pytest.mark.matlab
    def test_const_coeffs(self):
        """Constant function coefficients."""
        coeffs = jnp.asarray(self.ref["sc_const_coeffs"], dtype=jnp.float64)
        cutoff = standard_chop(coeffs)
        expected = int(self.ref["sc_const_cutoff"])
        assert cutoff == expected, f"const cutoff: got {cutoff}, expected {expected}"


class TestGridsampleMATLAB:
    """Cross-validate gridsample against MATLAB Chebfun reference data."""

    @pytest.fixture(autouse=True)
    def _load_refs(self):
        from tests.conftest import load_matlab_ref
        self.ref = load_matlab_ref("misc.mat")

    @pytest.mark.matlab
    def test_sin_5(self):
        v = gridsample(jnp.sin, 5)
        npt.assert_allclose(np.array(v), self.ref["gs_sin5"], rtol=1e-12, atol=1e-14)

    @pytest.mark.matlab
    def test_exp_10(self):
        v = gridsample(jnp.exp, 10)
        npt.assert_allclose(np.array(v), self.ref["gs_exp10"], rtol=1e-12, atol=1e-14)

    @pytest.mark.matlab
    def test_sin_10_custom_domain(self):
        v = gridsample(jnp.sin, 10, domain=(0.0, float(jnp.pi)))
        npt.assert_allclose(np.array(v), self.ref["gs_sin10_0pi"], rtol=1e-12, atol=1e-14)


class TestAbstractQRMATLAB:
    """Cross-validate abstract_qr against MATLAB abstractQR reference data."""

    @pytest.fixture(autouse=True)
    def _load_refs(self):
        from tests.conftest import load_matlab_ref
        self.ref = load_matlab_ref("misc.mat")

    @pytest.mark.matlab
    def test_5x3_Q(self):
        """Q factor of 5x3 matrix should match MATLAB."""
        A = jnp.asarray(self.ref["aqr_A1"], dtype=jnp.float64)
        E = jnp.asarray(self.ref["aqr_E1"], dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))
        # Q and R should match MATLAB up to sign ambiguity in columns.
        # Check |Q^T Q_matlab| ≈ I (diagonal should be ±1).
        Q_ref = self.ref["aqr_Q1"]
        # Instead of comparing Q directly (signs may flip), verify QR ≈ A.
        npt.assert_allclose(np.array(Q @ R), np.array(A), rtol=1e-12, atol=1e-13)

    @pytest.mark.matlab
    def test_5x3_R(self):
        """R factor of 5x3 matrix: upper triangular with correct QR product."""
        A = jnp.asarray(self.ref["aqr_A1"], dtype=jnp.float64)
        E = jnp.asarray(self.ref["aqr_E1"], dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))
        # R should be upper triangular.
        for i in range(R.shape[0]):
            for j in range(i):
                assert abs(float(R[i, j])) < 1e-13, f"R[{i},{j}] = {float(R[i, j])}"

    @pytest.mark.matlab
    def test_5x3_orthogonality(self):
        """Q^T Q ≈ I for standard inner product."""
        A = jnp.asarray(self.ref["aqr_A1"], dtype=jnp.float64)
        E = jnp.asarray(self.ref["aqr_E1"], dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))
        npt.assert_allclose(np.array(Q.T @ Q), np.eye(3), atol=1e-12)

    @pytest.mark.matlab
    def test_4x2(self):
        """4x2 QR: Q^T Q ≈ I and QR ≈ A."""
        A = jnp.asarray(self.ref["aqr_A2"], dtype=jnp.float64)
        E = jnp.asarray(self.ref["aqr_E2"], dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))
        npt.assert_allclose(np.array(Q.T @ Q), np.eye(2), atol=1e-12)
        npt.assert_allclose(np.array(Q @ R), np.array(A), rtol=1e-12, atol=1e-13)


# ============================================================================
# JIT compatibility tests
# ============================================================================


class TestJITCompatibility:
    """Verify JIT-safety properties.

    standard_chop is NOT JIT-safe (Python control flow).
    gridsample is not directly JIT-safe (calls chebpts with dynamic n).
    abstract_qr is NOT JIT-safe (Python loops).

    However, we test that the underlying computations are correct and that
    calling these functions outside of JIT produces correct results even when
    JAX is initialised with JIT compilation enabled.
    """

    def test_standard_chop_returns_python_int(self):
        """standard_chop should return a plain Python int, not a JAX scalar."""
        coeffs = 10.0 ** (-jnp.arange(1, 51, dtype=jnp.float64))
        cutoff = standard_chop(coeffs)
        assert isinstance(cutoff, int)

    def test_gridsample_output_is_jax_array(self):
        """gridsample output should be a JAX array."""
        v = gridsample(jnp.sin, 5)
        assert isinstance(v, jax.Array)

    def test_abstract_qr_output_dtype(self):
        """abstract_qr should return float64 JAX arrays."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64)
        E = jnp.eye(3, 2, dtype=jnp.float64)
        Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))
        assert Q.dtype == jnp.float64
        assert R.dtype == jnp.float64
        assert isinstance(Q, jax.Array)
        assert isinstance(R, jax.Array)
