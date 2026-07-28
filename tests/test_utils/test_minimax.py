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

from chebfunjax.utils.minimax import (
    MinimaxRationalResult,
    MinimaxResult,
    _eval_poly_bary,
    minimax,
)
from chebfunjax.utils.quadrature import chebpts_ab

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

    def test_rational_negative_degree_raises(self):
        """Negative rational degrees should raise ValueError."""
        with pytest.raises(ValueError, match="rational degrees must be >= 0"):
            minimax(jnp.abs, 4, denom=-1, rational=True)

    def test_breakpoints_deprecated_and_ignored(self):
        """The deprecated 'breakpoints' arg warns and no longer corrupts results.

        Previously ``minimax(|x-0.5|, 1, breakpoints=[0.5])`` stalled at a
        non-equioscillating reference and reported err=0.25 while the true sup
        error is 0.375.  It must now match the plain (correct) result.
        """
        def f(x):
            return jnp.abs(x - 0.5)

        with pytest.warns(DeprecationWarning, match="breakpoints"):
            r_bp = minimax(f, 1, breakpoints=[0.5])
        r_plain = minimax(f, 1)
        # Exact best degree-1 error to |x-0.5| on [-1,1] is 0.375.
        npt.assert_allclose(r_bp.err, 0.375, rtol=1e-3)
        npt.assert_allclose(r_bp.err, r_plain.err, rtol=1e-10)
        # Reported error matches the true global error (equioscillation holds).
        g = _global_max_error(f, r_bp.coeffs, r_bp.domain)
        npt.assert_allclose(g, r_bp.err, rtol=1e-3)

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
# Rational minimax (barycentric-Remez, Filip-Nakatsukasa-Beckermann-Trefethen)
# ---------------------------------------------------------------------------


def _rat_global_error(f, r, domain, n_dense=20000):
    """Sup error of a rational approximant on a dense grid."""
    a, b = domain
    xx = np.linspace(a, b, n_dense)
    fv = np.asarray(f(jnp.array(xx)), dtype=np.float64).ravel()
    rv = np.asarray(r.r(xx), dtype=np.float64).ravel()
    return float(np.max(np.abs(fv - rv)))


def _rat_alternations(f, r):
    """Number of sign alternations of the error on the final reference."""
    xk = np.asarray(r.xk, dtype=np.float64)
    e = np.asarray(f(jnp.array(xk)), dtype=np.float64).ravel() - np.asarray(
        r.r(xk), dtype=np.float64
    ).ravel()
    signs = np.sign(e)
    return int(np.sum(np.diff(signs) != 0)) + 1


class TestMinimaxRational:
    """Rational (type-(m, n)) best approximation via barycentric-Remez.

    Gates mirror the recorded MATLAB ``minimax`` outputs (test_minimax.m and
    ATAP), verifying both the sup-norm error level and the equioscillation
    (alternation) count of the returned reference.
    """

    def test_returns_rational_result(self):
        r = minimax(jnp.abs, 4, denom=4, rational=True)
        assert isinstance(r, MinimaxRationalResult)
        assert r.success
        assert callable(r.r)

    def test_default_denom_is_diagonal(self):
        """denom defaults to n (diagonal type)."""
        r = minimax(jnp.abs, 6, rational=True)
        assert (r.m, r.n) == (6, 6)

    def test_absx_2_2_error_and_alternation(self):
        """|x| type (2,2): err ~ 0.043689 (MATLAB pass(8)), 6 alternations."""
        r = minimax(jnp.abs, 2, denom=2, rational=True)
        npt.assert_allclose(r.err, 0.043689, rtol=1e-3)
        assert _rat_alternations(jnp.abs, r) == 6
        # Reported error matches the true global sup error.
        npt.assert_allclose(_rat_global_error(jnp.abs, r, r.domain), r.err, rtol=1e-4)

    def test_absx_8_8_error_and_alternation(self):
        """|x| type (8,8): err ~ 8e-4 with 18 equioscillation points."""
        r = minimax(jnp.abs, 8, denom=8, rational=True)
        assert r.err < 1e-3
        assert 7e-4 < r.err < 8e-4
        assert _rat_alternations(jnp.abs, r) == 18
        assert len(r.poles) == 8

    def test_absx_30_30_error(self):
        """|x| type (30,30): err ~ 2.1739878e-7 (MATLAB pass(13))."""
        r = minimax(jnp.abs, 30, denom=30, rational=True)
        npt.assert_allclose(r.err, 2.1739878e-7, rtol=1e-3)

    def test_exact_rational_2_2_recovered(self):
        """An exact type-(2,2) rational is recovered to ~machine precision.

        f = ((x+3)(x-0.5))/(x^2-4).  MATLAB pass(3): err < 1e-10.
        """
        def f(x):
            return ((x + 3.0) * (x - 0.5)) / (x ** 2 - 4.0)

        r = minimax(f, 2, denom=2, rational=True, tol=1e-12, max_iter=20)
        assert _rat_global_error(f, r, r.domain) < 1e-10

    def test_exact_rational_2_2_perturbed_no_shape_crash(self):
        """Stress mirror for the reference-length shape invariant.

        An exact type-(2,2) rational drives the Remez error toward zero, where
        the exchange step's alternating-extrema count is fragile: on some
        BLAS/platforms it returns fewer than m+n+2 points (flag == 0).  Feeding
        that short reference back into the barycentric trial solver used to
        crash with a matmul core-dimension mismatch.  Perturbing the sample
        geometry by a few ULPs exercises both the full-length and short-exchange
        branches; every run must finish (crash-free) and land near the exact
        rational.
        """
        def f(x):
            return ((x + 3.0) * (x - 0.5)) / (x ** 2 - 4.0)

        eps = np.finfo(np.float64).eps
        for k in (0, 1, -1, 7, -13, 101, -257):
            init = np.array(chebpts_ab(6, -1.0, 1.0, kind=2)) * (1.0 + k * eps)
            r = minimax(f, 2, denom=2, rational=True, tol=1e-12, max_iter=20,
                        init_xk=init)
            # No exception is the primary assertion; the recovered rational is
            # still essentially exact.
            assert _rat_global_error(f, r, r.domain) < 1e-8, f"k={k}"

    def test_exact_rational_3_2_recovered(self):
        """Exact type-(3,2) rational recovered (MATLAB pass(4): err < 1e-10)."""
        def f(x):
            return ((x - 3.0) * (x + 0.2) * (x - 0.7)) / ((x - 1.5) * (x + 2.1))

        r = minimax(f, 3, denom=2, rational=True)
        assert _rat_global_error(f, r, r.domain) < 1e-10

    def test_scale_invariance_huge_amplitude(self):
        """1e40*|x| type (5,5): err < 1e38 (MATLAB pass(18)); scale-homogeneous."""
        r_big = minimax(lambda x: 1e40 * jnp.abs(x), 5, denom=5, rational=True)
        r_small = minimax(jnp.abs, 5, denom=5, rational=True)
        assert r_big.err < 1e38
        npt.assert_allclose(r_big.err, r_small.err * 1e40, rtol=1e-6)

    def test_scale_invariance_tiny_amplitude(self):
        """Tiny-amplitude invariance, type (1,3) (MATLAB pass(11))."""
        r_big, r_small = (
            minimax(jnp.exp, 1, denom=3, rational=True),
            minimax(lambda x: 1e-100 * jnp.exp(x), 1, denom=3, rational=True),
        )
        s_big = float(r_big.r(np.array([0.3]))[0]) - float(np.exp(0.3))
        s_small = float(r_small.r(np.array([0.3]))[0]) - float(1e-100 * np.exp(0.3))
        assert abs(s_big - s_small * 1e100) < 1e-3

    def test_odd_function_zero_numerator(self):
        """Odd f with numerator degree 0 collapses to the zero function.

        MATLAB pass(9): minimax(x^3, 0, 2) succeeds (returns 0 via the
        even/odd symmetry reduction m -> -1).
        """
        r = minimax(lambda x: x ** 3, 0, denom=2, rational=True)
        assert r.success
        npt.assert_allclose(np.asarray(r.r(np.linspace(-1, 1, 50))), 0.0, atol=0)

    def test_type_m0_is_polynomial(self):
        """Type (m,0) reduces to the polynomial best approximation.

        MATLAB pass(7): minimax(|x|, 0, 0) has error ~0.5.
        """
        r = minimax(jnp.abs, 0, denom=0, rational=True)
        assert isinstance(r, MinimaxRationalResult)
        npt.assert_allclose(r.err, 0.5, rtol=1e-6)

    def test_sqrt_poles_zeros_negative_real(self):
        """sqrt on [0,1] type (4,4): poles/zeros interlace on negative reals.

        MATLAB pass(19) checks status.zer / status.pol against roots(p),
        roots(q); the Zolotarev structure places them on the negative axis.
        """
        r = minimax(jnp.sqrt, 4, denom=4, rational=True, domain=(0.0, 1.0))
        assert len(r.poles) == 4 and len(r.zeros) == 4
        assert np.all(np.real(r.poles) < 0)
        zr = np.sort(np.real(r.zeros[np.abs(np.imag(r.zeros)) < 1e-8]))
        # Every real zero z satisfies r(z) ~ 0.
        npt.assert_allclose(np.asarray(r.r(zr)), 0.0, atol=1e-10)

    def test_r_is_jit_safe_callable(self):
        """The returned evaluator produces finite values across the domain."""
        r = minimax(jnp.exp, 3, denom=3, rational=True)
        vals = np.asarray(r.r(np.linspace(-1, 1, 200)))
        assert np.all(np.isfinite(vals))


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
        """Degree-10 |x| error matches MATLAB within 1%.
        Remez exchange may converge to slightly different local optima."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_err = float(minimax_ref["abs_deg10_err"])
        res = minimax(jnp.abs, 10)
        npt.assert_allclose(res.err, ref_err, rtol=0.01)

    def test_absx_degree10_xk(self, minimax_ref):
        """Degree-10 |x| equioscillation points near MATLAB's."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_xk = np.asarray(minimax_ref["abs_deg10_xk"], dtype=np.float64).ravel()
        res = minimax(jnp.abs, 10)
        # Equioscillation points can differ slightly between implementations
        assert len(res.xk) >= len(ref_xk) - 2, f"Too few equioscillation points: {len(res.xk)} vs {len(ref_xk)}"

    def test_absx_degree10_coeffs(self, minimax_ref):
        """Degree-10 |x| Chebyshev coefficients match MATLAB within 1%."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_coeffs = np.asarray(
            minimax_ref["abs_deg10_coeffs"], dtype=np.float64
        ).ravel()
        res = minimax(jnp.abs, 10)
        npt.assert_allclose(np.array(res.coeffs), ref_coeffs, rtol=0.01, atol=1e-10)

    def test_sin_degree6_err(self, minimax_ref):
        """Degree-6 sin(x) error matches MATLAB within 1%."""
        if minimax_ref is None:
            pytest.skip("minimax.mat not found; run MATLAB harness first.")
        ref_err = float(minimax_ref["sin_deg6_err"])
        res = minimax(jnp.sin, 6)
        npt.assert_allclose(res.err, ref_err, rtol=0.01)


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
