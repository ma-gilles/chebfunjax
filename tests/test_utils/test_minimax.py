"""Tests for chebfunjax.utils.minimax -- Remez exchange / best polynomial approximation.

JAX contract:
  - ``minimax()`` construction: NOT JIT-safe (adaptive Remez loop with
    Python-level control flow).
  - ``result.coeffs``: a JAX float64 array, JIT-safe.

Mathematical properties verified:
  - Equioscillation: ``|f(xk) - p(xk)|`` is constant at all n+2 reference
    points with alternating sign.
  - Global maximum: the error at the reference equals the global max error.
  - Delta near zero: ``(err - |h|) / normf < tol`` after convergence.
  - Known best-approximation errors from the literature (Pachon & Trefethen 2009).

Reference:
    R. Pachon and L. N. Trefethen, "Barycentric-Remez algorithms for best
    polynomial approximation in the Chebfun system", BIT Numerical Mathematics,
    49:721-742, 2009.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from chebfunjax.utils.minimax import MinimaxResult, _eval_poly_bary, minimax

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _global_max_error(
    f,
    coeffs: jnp.ndarray,
    domain: tuple[float, float],
    n_dense: int = 100_000,
) -> float:
    """Estimate global max error on a dense grid."""
    a, b = domain
    xx = np.linspace(a, b, n_dense)
    p_vals = _eval_poly_bary(xx, np.array(coeffs), a, b)
    f_vals = np.asarray(f(jnp.array(xx)), dtype=np.float64).ravel()
    return float(np.max(np.abs(f_vals - p_vals)))


# ---------------------------------------------------------------------------
# Tier 1: Mathematical property tests (no MATLAB reference needed)
# ---------------------------------------------------------------------------

class TestMinimaxEquioscillation:
    """Verify equioscillation at all n+2 reference points."""

    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 20])
    def test_equioscillation_absx(self, n: int):
        """Degree-n best approx to |x|: all reference errors equal, alternating sign.

        The equioscillation theorem (Chebyshev's theorem) guarantees that the
        best polynomial approximant of degree n has exactly n+2 equioscillation
        points with alternating sign.  This is the defining property.
        """
        res = minimax(jnp.abs, n)
        xk = np.array(res.xk)
        a, b = res.domain
        fk = np.abs(xk)
        pk = _eval_poly_bary(xk, np.array(res.coeffs), a, b)
        errs = fk - pk
        abs_errs = np.abs(errs)

        # 1. All n+2 errors have the same absolute value (equioscillation)
        npt.assert_allclose(
            abs_errs,
            abs_errs[0],
            rtol=1e-10,
            err_msg=f"n={n}: error levels not equioscillating",
        )

        # 2. Signs alternate exactly
        signs = np.sign(errs)
        sign_changes = np.sum(np.diff(signs) != 0)
        assert sign_changes == n + 1, (
            f"n={n}: expected {n+1} sign changes, got {sign_changes}. "
            f"Signs: {signs}"
        )

        # 3. Correct number of reference points
        assert len(xk) == n + 2, (
            f"n={n}: expected {n+2} reference points, got {len(xk)}"
        )

    @pytest.mark.parametrize("n", [4, 10])
    def test_equioscillation_sin(self, n: int):
        """Degree-n best approximation to sin(x) on [-1, 1]."""
        res = minimax(jnp.sin, n)
        xk = np.array(res.xk)
        a, b = res.domain
        fk = np.sin(xk)
        pk = _eval_poly_bary(xk, np.array(res.coeffs), a, b)
        errs = fk - pk
        abs_errs = np.abs(errs)

        # Use absolute tolerance of 1e-13 in addition to relative, to handle
        # the case where the error itself is near machine precision (e.g. sin
        # at high degree where err ~ 1e-11 and floating-point noise dominates).
        npt.assert_allclose(
            abs_errs,
            abs_errs[0],
            rtol=1e-5,
            atol=1e-13,
            err_msg=f"sin(x) n={n}: error levels not equioscillating",
        )
        signs = np.sign(errs)
        sign_changes = np.sum(np.diff(signs) != 0)
        assert sign_changes == n + 1, (
            f"sin(x) n={n}: expected {n+1} sign changes, got {sign_changes}"
        )


class TestMinimaxGlobalError:
    """Verify that reported error matches the global max error."""

    @pytest.mark.parametrize("n", [4, 10, 20])
    def test_reported_error_matches_global(self, n: int):
        """The reported err should match the actual global max error."""
        res = minimax(jnp.abs, n)
        global_err = _global_max_error(jnp.abs, res.coeffs, res.domain)
        # Allow a small tolerance for the dense-grid approximation
        npt.assert_allclose(
            global_err,
            res.err,
            rtol=1e-5,
            err_msg=f"n={n}: reported err {res.err:.4e} vs global {global_err:.4e}",
        )

    def test_sin_global_error(self):
        """sin(x) degree-6 approximation: global error matches reported."""
        res = minimax(jnp.sin, 6)
        global_err = _global_max_error(jnp.sin, res.coeffs, res.domain)
        npt.assert_allclose(global_err, res.err, rtol=1e-5)


class TestMinimaxConvergence:
    """Verify convergence indicators (delta, iter)."""

    def test_delta_small(self):
        """delta/normf should be near zero after convergence."""
        res = minimax(jnp.abs, 10)
        assert res.delta < 1e-10, (
            f"delta/normf = {res.delta:.2e} is not near zero (algorithm did not "
            f"converge to best approximation)"
        )

    def test_iterations_bounded(self):
        """Algorithm should converge in at most 30 iterations for |x|."""
        for n in [4, 10, 20]:
            res = minimax(jnp.abs, n)
            assert res.iter <= 30, (
                f"n={n}: took {res.iter} iterations, expected <= 30"
            )

    def test_degree_zero(self):
        """Best degree-0 approximation to cos(x): should be mean-value constant."""
        res = minimax(jnp.cos, 0)
        assert len(res.coeffs) == 1
        assert len(res.xk) == 2
        # Best degree-0 approx to cos on [-1,1]: constant = (1 + cos(1)) / 2
        # The error should be (1 - cos(1)) / 2
        npt.assert_allclose(res.err, (1.0 - math.cos(1.0)) / 2.0, rtol=1e-8)

    def test_constant_function(self):
        """Best approximation to a constant is trivially the constant itself."""
        res = minimax(lambda x: 3.0 * jnp.ones_like(x), 5)
        assert res.err < 1e-14, f"Constant function: err={res.err:.2e} should be ~0"
        npt.assert_allclose(float(res.coeffs[0]), 3.0, rtol=1e-12)
        npt.assert_allclose(res.coeffs[1:], 0.0, atol=1e-12)


class TestMinimaxReturnType:
    """Verify the return type and structure of MinimaxResult."""

    def test_result_is_MinimaxResult(self):
        """Return value is a MinimaxResult dataclass."""
        res = minimax(jnp.abs, 4)
        assert isinstance(res, MinimaxResult)

    def test_coeffs_is_jax_array(self):
        """coeffs must be a JAX float64 array."""
        res = minimax(jnp.abs, 4)
        assert isinstance(res.coeffs, jnp.ndarray)
        assert res.coeffs.dtype == jnp.float64

    def test_xk_is_jax_array(self):
        """xk must be a JAX float64 array."""
        res = minimax(jnp.abs, 4)
        assert isinstance(res.xk, jnp.ndarray)
        assert res.xk.dtype == jnp.float64

    def test_sizes(self):
        """len(coeffs) == n+1, len(xk) == n+2."""
        for n in [0, 1, 3, 10]:
            res = minimax(jnp.sin, n)
            assert len(res.coeffs) == n + 1, (
                f"n={n}: expected len(coeffs)={n+1}, got {len(res.coeffs)}"
            )
            assert len(res.xk) == n + 2, (
                f"n={n}: expected len(xk)={n+2}, got {len(res.xk)}"
            )


class TestMinimaxDomain:
    """Verify domain support."""

    def test_custom_domain(self):
        """Approximation on [0, 1] should work the same as on [-1, 1]."""
        res = minimax(jnp.exp, 6, domain=(0.0, 1.0))
        global_err = _global_max_error(jnp.exp, res.coeffs, (0.0, 1.0))
        npt.assert_allclose(global_err, res.err, rtol=1e-5)
        # Equioscillation on [0, 1]
        xk = np.array(res.xk)
        fk = np.exp(xk)
        pk = _eval_poly_bary(xk, np.array(res.coeffs), 0.0, 1.0)
        errs = np.abs(fk - pk)
        # Use relative tolerance 1e-5 to accommodate floating-point noise at
        # the equioscillation level (errors ~4e-8).
        npt.assert_allclose(errs, errs[0], rtol=1e-5)

    def test_negative_domain(self):
        """Approximation on [-2, -0.5]."""
        def f(x):
            return jnp.sin(x)
        res = minimax(f, 4, domain=(-2.0, -0.5))
        global_err = _global_max_error(f, res.coeffs, (-2.0, -0.5))
        npt.assert_allclose(global_err, res.err, rtol=1e-5)


class TestMinimaxErrors:
    """Test that invalid inputs raise appropriate exceptions."""

    def test_rational_raises(self):
        """rational=True should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="rational=True"):
            minimax(jnp.abs, 4, rational=True)

    def test_negative_n_raises(self):
        """n < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            minimax(jnp.abs, -1)

    def test_invalid_domain_raises(self):
        """a >= b should raise ValueError."""
        with pytest.raises(ValueError, match="domain must satisfy a < b"):
            minimax(jnp.abs, 4, domain=(1.0, -1.0))

    def test_wrong_init_xk_length_raises(self):
        """init_xk with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="init_xk must have length"):
            minimax(jnp.abs, 4, init_xk=np.array([0.0, 0.5]))


# ---------------------------------------------------------------------------
# Tier 2: MATLAB golden reference comparisons
# ---------------------------------------------------------------------------

def _load_minimax_ref():
    """Load minimax.mat golden references (session-cached)."""
    from pathlib import Path

    import scipy.io

    path = Path(__file__).parent.parent / "references" / "minimax.mat"
    if not path.exists():
        return None
    return scipy.io.loadmat(str(path), squeeze_me=True)


@pytest.fixture(scope="session")
def minimax_ref():
    return _load_minimax_ref()


@pytest.mark.matlab
class TestMinimaxMATLAB:
    """MATLAB cross-validation tests.

    Require ``tests/references/minimax.mat`` (generated by
    ``matlab_harness/refs/minimax_refs.m``).
    """

    def test_absx_degree10_err(self, minimax_ref):
        """Degree-10 |x| error matches MATLAB to 1e-12."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_err = float(minimax_ref["abs_deg10_err"])
        res = minimax(jnp.abs, 10)
        npt.assert_allclose(res.err, ref_err, rtol=1e-12)

    def test_absx_degree10_xk(self, minimax_ref):
        """Degree-10 |x| equioscillation reference points match MATLAB."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_xk = np.asarray(minimax_ref["abs_deg10_xk"], dtype=np.float64).ravel()
        res = minimax(jnp.abs, 10)
        npt.assert_allclose(np.array(res.xk), ref_xk, atol=1e-10)

    def test_absx_degree10_coeffs(self, minimax_ref):
        """Degree-10 |x| Chebyshev coefficients match MATLAB to 1e-12."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_coeffs = np.asarray(
            minimax_ref["abs_deg10_coeffs"], dtype=np.float64
        ).ravel()
        res = minimax(jnp.abs, 10)
        npt.assert_allclose(np.array(res.coeffs), ref_coeffs, atol=1e-12)

    def test_sin_degree6_err(self, minimax_ref):
        """Degree-6 sin(x) error matches MATLAB."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_err = float(minimax_ref["sin_deg6_err"])
        res = minimax(jnp.sin, 6)
        npt.assert_allclose(res.err, ref_err, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tier 1 (continued): Known error values from the literature
# ---------------------------------------------------------------------------

class TestMinimaxLiteratureValues:
    """Verify against known best-approximation errors from the literature.

    The best approximation errors for |x| are tabulated in:
      R. Pachon and L. N. Trefethen, "Barycentric-Remez algorithms for best
      polynomial approximation in the Chebfun system", BIT Numerical
      Mathematics, 49:721-742, 2009.  Table 1.
    """

    # Table 1 from Pachon & Trefethen (2009) for minimax(|x|, n) on [-1, 1].
    # These are computed by the same Remez algorithm and verified to be
    # the true global minima via independent optimization.
    #   n  | err (from P&T Table 1)
    #  ----+--------------------
    #   2  | 0.125000000000000
    #   4  | 0.067620899277779
    #  10  | 0.027845118553947
    #  20  | 0.013986621698124
    ABSX_ERRORS = {
        2:  0.125000000000000,
        4:  0.067620899277779,
        10: 0.027845118553947,
        20: 0.013986621698124,
    }

    @pytest.mark.parametrize("n,expected_err", list(ABSX_ERRORS.items()))
    def test_absx_known_errors(self, n: int, expected_err: float):
        """Verify best-polynomial-approximation errors for |x| against table."""
        res = minimax(jnp.abs, n)
        npt.assert_allclose(
            res.err,
            expected_err,
            rtol=1e-8,
            err_msg=(
                f"n={n}: expected err~{expected_err:.6e}, got {res.err:.6e}"
            ),
        )
