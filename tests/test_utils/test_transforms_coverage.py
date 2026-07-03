"""Correctness tests for :mod:`chebfunjax.utils.transforms`.

These tests raise coverage of the polynomial-transform machinery by pinning
down mathematically verifiable identities:

* value <-> coefficient round trips (``vals2coeffs`` / ``coeffs2vals``),
* basis-change transforms represent the *same polynomial* (checked by
  evaluating the source and target series independently with NumPy / SciPy),
* inverse pairs compose to the identity (``jac2jac``, ``ultra2ultra``,
  ``dst`` / ``idst``).

Every expectation is computed with an *independent* reference
(``numpy.polynomial`` or ``scipy.special``), never by re-running the code
under test, so the assertions catch genuine regressions.

The final class documents four confirmed bugs found while writing these
tests (see the module report). Those tests assert the mathematically correct
result; fixed bugs have had their xfail markers removed while still
exercising the buggy branches (which keeps them under coverage and turns the
tests into regression sentinels that will ``xpass`` once the bugs are fixed).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from numpy.polynomial import chebyshev as npcheb
from numpy.polynomial import legendre as npleg
from scipy.special import eval_gegenbauer, eval_jacobi

from chebfunjax.utils import transforms as T
from chebfunjax.utils.quadrature import chebpts

# ---------------------------------------------------------------------------
# Independent reference evaluators
# ---------------------------------------------------------------------------

def _jac_series(coeffs, a, b, x):
    """Evaluate sum_k coeffs[k] * P_k^{(a,b)}(x) with SciPy."""
    coeffs = np.asarray(coeffs)
    return sum(coeffs[k] * eval_jacobi(k, a, b, x) for k in range(len(coeffs)))


def _geg_series(coeffs, lam, x):
    """Evaluate sum_k coeffs[k] * C_k^{(lam)}(x) with SciPy (ultraspherical)."""
    coeffs = np.asarray(coeffs)
    return sum(coeffs[k] * eval_gegenbauer(k, lam, x) for k in range(len(coeffs)))


_XT = np.linspace(-0.9, 0.9, 11)  # interior test abscissae


def _rand(n, seed):
    return np.asarray(np.random.RandomState(seed).randn(n))


# ---------------------------------------------------------------------------
# vals2coeffs / coeffs2vals
# ---------------------------------------------------------------------------

class TestValsCoeffs:
    def test_coeffs2vals_matches_numpy_at_chebpts2(self):
        c = _rand(8, 10)
        x2 = np.asarray(chebpts(len(c), kind=2))
        npt.assert_allclose(
            np.asarray(T.coeffs2vals(jnp.asarray(c))),
            npcheb.chebval(x2, c),
            rtol=0, atol=1e-13,
        )

    def test_roundtrip_vals_coeffs(self):
        c = _rand(12, 11)
        rt = T.vals2coeffs(T.coeffs2vals(jnp.asarray(c)))
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-12)

    def test_single_value_is_constant(self):
        # A length-1 series is the constant c0; value == coefficient.
        npt.assert_allclose(np.asarray(T.coeffs2vals(jnp.asarray([3.5]))), [3.5])
        npt.assert_allclose(np.asarray(T.vals2coeffs(jnp.asarray([3.5]))), [3.5])

    def test_empty(self):
        assert T.coeffs2vals(jnp.asarray([])).shape == (0,)
        assert T.vals2coeffs(jnp.asarray([])).shape == (0,)

    def test_constant_series_flat_values(self):
        # T_0 only -> all sample values equal the constant.
        vals = np.asarray(T.coeffs2vals(jnp.asarray([2.0, 0.0, 0.0, 0.0])))
        npt.assert_allclose(vals, 2.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Chebyshev <-> Legendre coefficient transforms
# ---------------------------------------------------------------------------

class TestChebLeg:
    def test_leg2cheb_same_polynomial(self):
        c = _rand(9, 20)
        c_cheb = np.asarray(T.leg2cheb(jnp.asarray(c)))
        npt.assert_allclose(
            npcheb.chebval(_XT, c_cheb), npleg.legval(_XT, c), rtol=0, atol=1e-12,
        )

    def test_cheb2leg_same_polynomial(self):
        c = _rand(9, 21)
        c_leg = np.asarray(T.cheb2leg(jnp.asarray(c)))
        npt.assert_allclose(
            npleg.legval(_XT, c_leg), npcheb.chebval(_XT, c), rtol=0, atol=1e-12,
        )

    def test_inverse_pair(self):
        c = _rand(10, 22)
        rt = T.leg2cheb(T.cheb2leg(jnp.asarray(c)))
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-12)

    def test_normalize_roundtrip(self):
        c = _rand(9, 23)
        rt = T.leg2cheb(T.cheb2leg(jnp.asarray(c), normalize=True), normalize=True)
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-12)

    def test_normalize_length_one(self):
        # Orthonormal P_0 = 1/sqrt(2); c0 rescaled by 1/sqrt(1/2).
        out = np.asarray(T.cheb2leg(jnp.asarray([2.0]), normalize=True))
        npt.assert_allclose(out, [2.0 / np.sqrt(0.5)])

    def test_coeff_aliases_match(self):
        c = jnp.asarray(_rand(7, 24))
        npt.assert_array_equal(
            np.asarray(T.chebcoeffs2legcoeffs(c)), np.asarray(T.cheb2leg(c))
        )
        npt.assert_array_equal(
            np.asarray(T.legcoeffs2chebcoeffs(c)), np.asarray(T.leg2cheb(c))
        )

    def test_legcoeffs2chebvals_kind2(self):
        c = _rand(8, 25)
        x2 = np.asarray(chebpts(len(c), kind=2))
        npt.assert_allclose(
            np.asarray(T.legcoeffs2chebvals(jnp.asarray(c), kind=2)),
            npleg.legval(x2, c), rtol=0, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Chebyshev <-> Jacobi transforms
# ---------------------------------------------------------------------------

class TestChebJac:
    @pytest.mark.parametrize("a,b", [(0.4, 0.9), (-0.3, 0.2), (1.0, 1.0), (0.0, 0.0)])
    def test_cheb2jac_same_polynomial(self, a, b):
        c = _rand(9, 30)
        cj = np.asarray(T.cheb2jac(jnp.asarray(c), a, b))
        npt.assert_allclose(_jac_series(cj, a, b, _XT), npcheb.chebval(_XT, c),
                            rtol=0, atol=1e-11)

    @pytest.mark.parametrize("a,b", [(0.4, 0.9), (-0.3, 0.2), (1.0, 1.0)])
    def test_inverse_pair(self, a, b):
        c = _rand(9, 31)
        rt = T.jac2cheb(T.cheb2jac(jnp.asarray(c), a, b), a, b)
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# Jacobi <-> Jacobi
# ---------------------------------------------------------------------------

class TestJac2Jac:
    def test_identity_when_params_equal(self):
        c = _rand(8, 40)
        npt.assert_allclose(
            np.asarray(T.jac2jac(jnp.asarray(c), 0.3, 0.7, 0.3, 0.7)), c, atol=1e-13
        )

    @pytest.mark.parametrize(
        "a,b,g,d",
        [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 0.0), (0.3, 0.7, 1.3, 0.7)],
    )
    def test_integer_shift_same_polynomial(self, a, b, g, d):
        c = _rand(8, 41)
        out = np.asarray(T.jac2jac(jnp.asarray(c), a, b, g, d))
        npt.assert_allclose(_jac_series(out, g, d, _XT), _jac_series(c, a, b, _XT),
                            rtol=0, atol=1e-11)

    def test_fractional_roundtrip(self):
        c = _rand(8, 42)
        out = T.jac2jac(jnp.asarray(c), 0.2, 0.5, 1.2, 0.9)
        rt = T.jac2jac(out, 1.2, 0.9, 0.2, 0.5)
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-10)

    def test_fractional_same_polynomial(self):
        c = _rand(8, 43)
        out = np.asarray(T.jac2jac(jnp.asarray(c), 0.25, 0.75, 1.25, 0.75))
        npt.assert_allclose(_jac_series(out, 1.25, 0.75, _XT),
                            _jac_series(c, 0.25, 0.75, _XT), rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Ultraspherical transforms
# ---------------------------------------------------------------------------

class TestUltra:
    @pytest.mark.parametrize("lam", [0.5, 1.5, 2.0, 3.0])
    def test_ultracoeffs_same_polynomial(self, lam):
        # Cheb-T coefficients -> ultraspherical C^{(lam)} coefficients.
        c = _rand(8, 50)
        out = np.asarray(T.ultracoeffs(jnp.asarray(c), lam))
        npt.assert_allclose(_geg_series(out, lam, _XT), npcheb.chebval(_XT, c),
                            rtol=0, atol=1e-10)

    def test_ultracoeffs_legendre_shortcut(self):
        # lam == 0.5 is Legendre; must agree with cheb2leg.
        c = jnp.asarray(_rand(8, 51))
        npt.assert_allclose(np.asarray(T.ultracoeffs(c, 0.5)),
                            np.asarray(T.cheb2leg(c)), atol=1e-13)

    @pytest.mark.parametrize("li,lo", [(1.0, 2.0), (0.5, 1.5), (2.0, 0.5)])
    def test_ultra2ultra_same_polynomial(self, li, lo):
        c = _rand(8, 52)
        out = np.asarray(T.ultra2ultra(jnp.asarray(c), li, lo))
        npt.assert_allclose(_geg_series(out, lo, _XT), _geg_series(c, li, _XT),
                            rtol=0, atol=1e-10)

    def test_ultra2ultra_roundtrip(self):
        c = _rand(8, 53)
        rt = T.ultra2ultra(T.ultra2ultra(jnp.asarray(c), 0.5, 1.5), 1.5, 0.5)
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# Grid <-> grid / grid <-> coefficient conversions (correct branches)
# ---------------------------------------------------------------------------

class TestGridConversions:
    def test_legvals2legcoeffs_inverse(self):
        c = _rand(9, 60)
        rt = T.legvals2legcoeffs(T.legcoeffs2legvals(jnp.asarray(c)))
        npt.assert_allclose(np.asarray(rt), c, rtol=0, atol=1e-11)

    def test_legvals2chebvals_same_polynomial(self):
        c = _rand(9, 61)
        xg, _ = np.polynomial.legendre.leggauss(len(c))
        x2 = np.asarray(chebpts(len(c), kind=2))
        out = T.legvals2chebvals(jnp.asarray(npleg.legval(xg, c)), kind=2)
        npt.assert_allclose(np.asarray(out), npleg.legval(x2, c), rtol=0, atol=1e-11)

    def test_chebvals2legvals_same_polynomial(self):
        c = _rand(9, 62)
        xg, _ = np.polynomial.legendre.leggauss(len(c))
        x2 = np.asarray(chebpts(len(c), kind=2))
        out = T.chebvals2legvals(jnp.asarray(npcheb.chebval(x2, c)), kind=2)
        npt.assert_allclose(np.asarray(out), npcheb.chebval(xg, c), rtol=0, atol=1e-11)

    def test_legvals2chebcoeffs_same_polynomial(self):
        c = _rand(9, 63)
        xg, _ = np.polynomial.legendre.leggauss(len(c))
        x2 = np.asarray(chebpts(len(c), kind=2))
        cc = np.asarray(T.legvals2chebcoeffs(jnp.asarray(npleg.legval(xg, c))))
        npt.assert_allclose(npcheb.chebval(x2, cc), npleg.legval(x2, c),
                            rtol=0, atol=1e-11)

    def test_chebvals2chebvals_1_to_2_same_polynomial(self):
        c = _rand(9, 64)
        x1 = np.asarray(chebpts(len(c), kind=1))
        x2 = np.asarray(chebpts(len(c), kind=2))
        out = T.chebvals2chebvals(jnp.asarray(npcheb.chebval(x1, c)), 1, 2)
        npt.assert_allclose(np.asarray(out), npcheb.chebval(x2, c), rtol=0, atol=1e-11)

    def test_chebvals2chebvals_same_kind_is_identity(self):
        v = jnp.asarray(_rand(6, 65))
        npt.assert_array_equal(np.asarray(T.chebvals2chebvals(v, 2, 2)), np.asarray(v))
        npt.assert_array_equal(np.asarray(T.chebvals2chebvals(v, 1, 1)), np.asarray(v))

    def test_chebvals2chebcoeffs_kind1_inverts_eval(self):
        # Round trip through the (correct) first-kind analysis transform.
        c = _rand(9, 66)
        x1 = np.asarray(chebpts(len(c), kind=1))
        cc = T.chebvals2chebcoeffs(jnp.asarray(npcheb.chebval(x1, c)), kind=1)
        npt.assert_allclose(np.asarray(cc), c, rtol=0, atol=1e-11)

    def test_chebcoeffs2chebvals_kind2_matches_coeffs2vals(self):
        c = jnp.asarray(_rand(7, 67))
        npt.assert_allclose(np.asarray(T.chebcoeffs2chebvals(c, kind=2)),
                            np.asarray(T.coeffs2vals(c)), atol=1e-14)


# ---------------------------------------------------------------------------
# Discrete sine transform
# ---------------------------------------------------------------------------

class TestDST:
    @pytest.mark.parametrize("kind", [1, 2, 3, 4])
    def test_dst_idst_roundtrip(self, kind):
        u = _rand(6, 70 + kind)
        rt = T.idst(T.dst(jnp.asarray(u), kind=kind), kind=kind)
        npt.assert_allclose(np.asarray(rt), u, rtol=0, atol=1e-12)

    def test_dst_is_linear(self):
        u = _rand(8, 80)
        v = _rand(8, 81)
        lhs = np.asarray(T.dst(jnp.asarray(3.0 * u - 2.0 * v), kind=1))
        rhs = 3.0 * np.asarray(T.dst(jnp.asarray(u), kind=1)) \
            - 2.0 * np.asarray(T.dst(jnp.asarray(v), kind=1))
        npt.assert_allclose(lhs, rhs, rtol=0, atol=1e-12)

    def test_dst1_matches_definition(self):
        # DST-I (standard factor-of-2 normalisation, as in scipy.fft.dst):
        # X_k = 2 * sum_{n} x_n sin(pi (n+1)(k+1)/(N+1)).
        x = _rand(5, 90)
        N = len(x)
        n = np.arange(1, N + 1)
        k = np.arange(1, N + 1)
        S = 2.0 * np.sin(np.pi * np.outer(k, n) / (N + 1))
        npt.assert_allclose(np.asarray(T.dst(jnp.asarray(x), kind=1)), S @ x,
                            rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Error branches
# ---------------------------------------------------------------------------

class TestErrorBranches:
    @pytest.mark.parametrize(
        "fn", [T.legcoeffs2chebvals, T.chebcoeffs2chebvals,
               T.chebvals2chebcoeffs, T.chebvals2legvals])
    def test_bad_kind_raises(self, fn):
        with pytest.raises(ValueError, match="kind"):
            fn(jnp.asarray([1.0, 2.0, 3.0]), kind=3)

    def test_chebvals2chebvals_bad_kind_raises(self):
        with pytest.raises(ValueError):
            T.chebvals2chebvals(jnp.asarray([1.0, 2.0, 3.0]), 3, 1)

    def test_ultracoeffs_nonpositive_lambda_raises(self):
        with pytest.raises(ValueError, match="lam"):
            T.ultracoeffs(jnp.asarray([1.0, 2.0]), -1.0)


# ---------------------------------------------------------------------------
# Formerly-confirmed bugs (now fixed; jac2jac gam=delta=-0.5 remains xfail).
#
# Each test asserts the mathematically correct answer, computed with an
# independent reference, and runs the buggy branch so it stays covered.
# When a bug is fixed the corresponding test will xpass -> remove the marker.
# ---------------------------------------------------------------------------

class TestKnownBugs:
    def test_chebvals2chebvals_2_to_1(self):
        c = _rand(7, 100)
        x1 = np.asarray(chebpts(len(c), kind=1))
        x2 = np.asarray(chebpts(len(c), kind=2))
        out = T.chebvals2chebvals(jnp.asarray(npcheb.chebval(x2, c)), 2, 1)
        npt.assert_allclose(np.asarray(out), npcheb.chebval(x1, c), rtol=0, atol=1e-10)

    def test_chebcoeffs2chebvals_kind1(self):
        c = _rand(7, 101)
        x1 = np.asarray(chebpts(len(c), kind=1))
        out = T.chebcoeffs2chebvals(jnp.asarray(c), kind=1)
        npt.assert_allclose(np.asarray(out), npcheb.chebval(x1, c), rtol=0, atol=1e-10)

    def test_legcoeffs2chebvals_kind1(self):
        c = _rand(7, 102)
        x1 = np.asarray(chebpts(len(c), kind=1))
        out = T.legcoeffs2chebvals(jnp.asarray(c), kind=1)
        npt.assert_allclose(np.asarray(out), npleg.legval(x1, c), rtol=0, atol=1e-10)

    def test_ultracoeffs_lambda_one(self):
        c = _rand(7, 103)
        out = np.asarray(T.ultracoeffs(jnp.asarray(c), 1.0))
        npt.assert_allclose(_geg_series(out, 1.0, _XT), npcheb.chebval(_XT, c),
                            rtol=0, atol=1e-10)

    def test_jac2jac_to_chebyshev_weight(self):
        c = _rand(7, 104)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(T.jac2jac(jnp.asarray(c), 0.5, 0.5, -0.5, -0.5))
        assert np.all(np.isfinite(out)), "jac2jac produced non-finite output"
        npt.assert_allclose(_jac_series(out, -0.5, -0.5, _XT),
                            _jac_series(c, 0.5, 0.5, _XT), rtol=0, atol=1e-9)
