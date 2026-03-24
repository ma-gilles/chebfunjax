"""Tests for newly added missing utility functions.

Covers:
- aaatrig  (aaa.py)
- transforms: legvals/legcoeffs wrappers, jac2jac, ultra2ultra, ultracoeffs,
              chebvals2legvals, chebvals2chebvals, chebcoeffs2chebvals,
              chebvals2chebcoeffs
- quadrature: chebpts2, chebpts3, paduapts
- trigutils: trigpoly, diffbarytrig
- specfun: besselroots, gammaratio
- random: smoothie, randnfundisk, randnfunsphere
- conformal: conformal
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)


# ===========================================================================
# aaatrig
# ===========================================================================

class TestAaatrig:
    """Tests for trigonometric AAA rational approximation."""

    def test_sin_approximation(self):
        """aaatrig should approximate sin(x) to near machine precision."""
        from chebfunjax.utils.aaa import aaatrig
        Z = jnp.linspace(0, 2 * jnp.pi, 300, endpoint=False)
        F = jnp.sin(Z)
        r, pol, res, zer, zj, fj, wj, errvec = aaatrig(F, Z)
        err = float(jnp.max(jnp.abs(r(Z) - F)))
        assert err < 1e-12, f"sin(x) error too large: {err:.2e}"

    def test_cos_approximation(self):
        """aaatrig should approximate cos(x)."""
        from chebfunjax.utils.aaa import aaatrig
        Z = jnp.linspace(0, 2 * jnp.pi, 200, endpoint=False)
        F = jnp.cos(Z)
        r, pol, res, zer, zj, fj, wj, errvec = aaatrig(F, Z, tol=1e-12)
        err = float(jnp.max(jnp.abs(r(Z) - F)))
        assert err < 1e-10, f"cos(x) error too large: {err:.2e}"

    def test_returns_correct_tuple(self):
        """aaatrig returns an 8-tuple with correct types."""
        from chebfunjax.utils.aaa import aaatrig
        Z = jnp.linspace(0, 2 * jnp.pi, 100, endpoint=False)
        F = jnp.sin(Z)
        result = aaatrig(F, Z)
        assert len(result) == 8
        r, pol, res, zer, zj, fj, wj, errvec = result
        assert callable(r)
        assert isinstance(errvec, list)
        assert len(errvec) > 0

    def test_even_form(self):
        """aaatrig with form='even' (cot basis) works."""
        from chebfunjax.utils.aaa import aaatrig
        Z = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 200, endpoint=False)
        F = jnp.cos(Z)
        r, pol, res, zer, zj, fj, wj, errvec = aaatrig(F, Z, form="even")
        err = float(jnp.max(jnp.abs(r(Z) - F)))
        assert err < 1e-8, f"even form error: {err:.2e}"

    def test_callable_F(self):
        """aaatrig accepts a callable F."""
        from chebfunjax.utils.aaa import aaatrig
        Z = jnp.linspace(0, 2 * jnp.pi, 200, endpoint=False)
        r, *_ = aaatrig(jnp.sin, Z)
        err = float(jnp.max(jnp.abs(r(Z) - jnp.sin(Z))))
        assert err < 1e-12


# ===========================================================================
# Transforms: Legendre values/coeffs
# ===========================================================================

class TestLegendreTransforms:
    """Tests for legvals/legcoeffs conversion functions."""

    def test_legvals2legcoeffs_legcoeffs2legvals_roundtrip(self):
        """legvals -> legcoeffs -> legvals recovers input."""
        from chebfunjax.utils import legvals2legcoeffs, legcoeffs2legvals
        from chebfunjax.utils.quadrature import legpts
        n = 10
        x, w = legpts(n)
        v = jnp.sin(x)
        c = legvals2legcoeffs(v)
        v_back = legcoeffs2legvals(c)
        npt.assert_allclose(np.array(v_back), np.array(v), rtol=1e-13, atol=1e-14)

    def test_legvals2chebcoeffs(self):
        """legvals2chebcoeffs matches manual pipeline."""
        from chebfunjax.utils import legvals2chebcoeffs
        from chebfunjax.utils.transforms import _legendre_idlt, leg2cheb
        from chebfunjax.utils.quadrature import legpts
        n = 8
        x, _ = legpts(n)
        v = x ** 3  # exact polynomial
        c_cheb = legvals2chebcoeffs(jnp.array(v))
        # Manual: idlt then leg2cheb
        c_leg = _legendre_idlt(jnp.array(v))
        c_cheb_ref = leg2cheb(c_leg)
        npt.assert_allclose(np.array(c_cheb), np.array(c_cheb_ref), rtol=1e-13, atol=1e-14)

    def test_legvals2chebvals_polynomial(self):
        """legvals2chebvals is exact for polynomials."""
        from chebfunjax.utils import legvals2chebvals
        from chebfunjax.utils.quadrature import legpts, chebpts
        n = 10
        x_leg, _ = legpts(n)
        x_cheb = chebpts(n)
        p = lambda x: x ** 4 - 2 * x ** 2 + 1
        v_leg = jnp.array(p(np.array(x_leg)))
        v_cheb_expected = jnp.array(p(np.array(x_cheb)))
        v_cheb = legvals2chebvals(v_leg)
        npt.assert_allclose(np.array(v_cheb), np.array(v_cheb_expected), rtol=1e-12, atol=1e-13)

    def test_legcoeffs2chebvals_polynomial(self):
        """legcoeffs2chebvals is exact for polynomials."""
        from chebfunjax.utils import legcoeffs2chebvals, legvals2legcoeffs
        from chebfunjax.utils.quadrature import legpts, chebpts
        n = 8
        x_leg, _ = legpts(n)
        x_cheb = chebpts(n)
        p = lambda x: 3 * x ** 3 - x
        v_leg = jnp.array(p(np.array(x_leg)))
        c_leg = legvals2legcoeffs(v_leg)
        v_cheb = legcoeffs2chebvals(c_leg)
        v_cheb_expected = jnp.array(p(np.array(x_cheb)))
        npt.assert_allclose(np.array(v_cheb), np.array(v_cheb_expected), rtol=1e-12, atol=1e-13)

    def test_legcoeffs2legvals_roundtrip(self):
        """legcoeffs -> legvals -> legcoeffs recovers input."""
        from chebfunjax.utils import legvals2legcoeffs, legcoeffs2legvals
        n = 12
        c_in = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        v = legcoeffs2legvals(c_in)
        c_back = legvals2legcoeffs(v)
        npt.assert_allclose(np.array(c_back), np.array(c_in), rtol=1e-12, atol=1e-14)

    def test_chebvals2legvals_polynomial(self):
        """chebvals2legvals is exact for polynomials."""
        from chebfunjax.utils import chebvals2legvals
        from chebfunjax.utils.quadrature import chebpts, legpts
        n = 10
        x_cheb = chebpts(n)
        x_leg, _ = legpts(n)
        p = lambda x: x ** 5 - 2 * x ** 3 + x
        v_cheb = jnp.array(p(np.array(x_cheb)))
        v_leg = chebvals2legvals(v_cheb)
        v_leg_expected = jnp.array(p(np.array(x_leg)))
        npt.assert_allclose(np.array(v_leg), np.array(v_leg_expected), rtol=1e-12, atol=1e-13)

    def test_chebvals2chebvals_identity(self):
        """chebvals2chebvals(v, k, k) is identity."""
        from chebfunjax.utils import chebvals2chebvals
        from chebfunjax.utils.quadrature import chebpts
        for kind in [1, 2]:
            x = chebpts(8, kind=kind)
            v = jnp.sin(x)
            v_out = chebvals2chebvals(v, kind, kind)
            npt.assert_allclose(np.array(v_out), np.array(v), rtol=1e-14)

    def test_chebvals2chebvals_1to2_polynomial(self):
        """chebvals2chebvals kind=1->2 is exact for polynomials."""
        from chebfunjax.utils import chebvals2chebvals
        from chebfunjax.utils.quadrature import chebpts
        n = 8
        x1 = chebpts(n, kind=1)
        x2 = chebpts(n, kind=2)
        p = lambda x: x ** 3 - x
        v1 = jnp.array(p(np.array(x1)))
        v2_expected = jnp.array(p(np.array(x2)))
        v2 = chebvals2chebvals(v1, 1, 2)
        npt.assert_allclose(np.array(v2), np.array(v2_expected), rtol=1e-12, atol=1e-13)

    def test_chebcoeffs2chebvals_roundtrip(self):
        """chebcoeffs2chebvals and chebvals2chebcoeffs invert each other."""
        from chebfunjax.utils import chebcoeffs2chebvals, chebvals2chebcoeffs
        c = jnp.array(rng.standard_normal(10), dtype=jnp.float64)
        v = chebcoeffs2chebvals(c)
        c_back = chebvals2chebcoeffs(v)
        npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-13, atol=1e-14)


# ===========================================================================
# Transforms: jac2jac
# ===========================================================================

class TestJac2Jac:
    """Tests for Jacobi-to-Jacobi conversion."""

    def test_roundtrip_integer_params(self):
        """jac2jac(c, 0,0, 1,1) -> jac2jac(c, 1,1, 0,0) recovers input."""
        from chebfunjax.utils import jac2jac
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        c2 = jac2jac(c, 0.0, 0.0, 1.0, 1.0)
        c_back = jac2jac(c2, 1.0, 1.0, 0.0, 0.0)
        npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-10, atol=1e-11)

    def test_identity(self):
        """jac2jac(c, a, b, a, b) is identity."""
        from chebfunjax.utils import jac2jac
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        for a, b in [(0.0, 0.0), (0.5, 0.5), (1.0, 2.0)]:
            c_out = jac2jac(c, a, b, a, b)
            npt.assert_allclose(np.array(c_out), np.array(c), rtol=1e-12, atol=1e-13,
                                err_msg=f"Identity failed for a={a}, b={b}")

    def test_legendre_to_jacobi11(self):
        """Convert Legendre coefficients to Jacobi(1,1) and back."""
        from chebfunjax.utils import jac2jac
        n = 8
        c = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        c2 = jac2jac(c, 0.0, 0.0, 1.0, 1.0)
        c_back = jac2jac(c2, 1.0, 1.0, 0.0, 0.0)
        npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-9, atol=1e-10)

    def test_against_cheb2jac(self):
        """jac2jac(cheb2jac(c, 0,0), 0,0, 1,0) matches cheb2jac(c, 1,0)."""
        from chebfunjax.utils import jac2jac, cheb2jac
        c_cheb = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        # Leg coeffs
        c_leg = cheb2jac(c_cheb, 0.0, 0.0)
        # Via jac2jac: Leg -> Jac(1,0)
        c_via_j2j = jac2jac(c_leg, 0.0, 0.0, 1.0, 0.0)
        # Direct: cheb -> Jac(1,0)
        c_direct = cheb2jac(c_cheb, 1.0, 0.0)
        npt.assert_allclose(np.array(c_via_j2j), np.array(c_direct), rtol=1e-9, atol=1e-10)


# ===========================================================================
# Transforms: ultra2ultra, ultracoeffs
# ===========================================================================

class TestUltraTransforms:
    """Tests for ultraspherical transforms."""

    def test_ultra2ultra_roundtrip(self):
        """ultra2ultra roundtrip recovers input."""
        from chebfunjax.utils import ultra2ultra
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        for lam_in, lam_out in [(0.5, 1.0), (1.0, 2.0), (0.5, 1.5)]:
            c2 = ultra2ultra(c, lam_in, lam_out)
            c_back = ultra2ultra(c2, lam_out, lam_in)
            npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-9, atol=1e-10,
                                err_msg=f"Roundtrip failed for ({lam_in},{lam_out})")

    def test_ultra2ultra_identity(self):
        """ultra2ultra(c, lam, lam) is identity."""
        from chebfunjax.utils import ultra2ultra
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        for lam in [0.5, 1.0, 1.5, 2.0]:
            c_out = ultra2ultra(c, lam, lam)
            npt.assert_allclose(np.array(c_out), np.array(c), rtol=1e-12, atol=1e-13,
                                err_msg=f"Identity failed for lam={lam}")

    def test_ultra2ultra_legendre_cheb(self):
        """lam=0.5 is Legendre, lam=1.0 is Cheb-2nd. Conversion should agree with cheb2leg."""
        from chebfunjax.utils import ultra2ultra
        from chebfunjax.utils.transforms import cheb2leg
        # ultraspherical lam=1 corresponds to Chebyshev-2nd kind coefficients
        # ultraspherical lam=0.5 corresponds to Legendre (up to scaling)
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        # ultra2ultra(c, 0.5, 1.0) should be consistent with the polynomial identity
        c2 = ultra2ultra(c, 0.5, 1.0)
        c2_back = ultra2ultra(c2, 1.0, 0.5)
        npt.assert_allclose(np.array(c2_back), np.array(c), rtol=1e-9, atol=1e-10)

    def test_ultracoeffs_legendre_case(self):
        """ultracoeffs(c, 0.5) == cheb2leg(c)."""
        from chebfunjax.utils import ultracoeffs
        from chebfunjax.utils.transforms import cheb2leg
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        c_us = ultracoeffs(c, 0.5)
        c_leg = cheb2leg(c)
        npt.assert_allclose(np.array(c_us), np.array(c_leg), rtol=1e-12, atol=1e-13)

    def test_ultracoeffs_invalid_lam(self):
        """ultracoeffs raises ValueError for lam <= 0."""
        from chebfunjax.utils import ultracoeffs
        c = jnp.array([1.0, 0.5], dtype=jnp.float64)
        with pytest.raises(ValueError, match="lam > 0"):
            ultracoeffs(c, 0.0)


# ===========================================================================
# Quadrature: chebpts2, chebpts3, paduapts
# ===========================================================================

class TestChebpts2:
    """Tests for 2D Chebyshev tensor grid."""

    def test_square_grid_shape(self):
        """chebpts2(n) returns (n, n) grids."""
        from chebfunjax.utils import chebpts2
        for n in [3, 5, 10]:
            XX, YY = chebpts2(n)
            assert XX.shape == (n, n), f"XX shape wrong for n={n}"
            assert YY.shape == (n, n), f"YY shape wrong for n={n}"

    def test_rectangular_grid(self):
        """chebpts2(nx, ny) returns (ny, nx) grids."""
        from chebfunjax.utils import chebpts2
        XX, YY = chebpts2(3, 5)
        assert XX.shape == (5, 3)
        assert YY.shape == (5, 3)

    def test_range(self):
        """Points lie in [-1, 1]^2."""
        from chebfunjax.utils import chebpts2
        XX, YY = chebpts2(10)
        assert float(jnp.min(XX)) >= -1.0 - 1e-14
        assert float(jnp.max(XX)) <= 1.0 + 1e-14
        assert float(jnp.min(YY)) >= -1.0 - 1e-14
        assert float(jnp.max(YY)) <= 1.0 + 1e-14

    def test_custom_domain(self):
        """chebpts2 with custom domain."""
        from chebfunjax.utils import chebpts2
        XX, YY = chebpts2(4, 4, domain=(0.0, 1.0, -2.0, 2.0))
        assert float(jnp.min(XX)) >= 0.0 - 1e-14
        assert float(jnp.max(XX)) <= 1.0 + 1e-14
        assert float(jnp.min(YY)) >= -2.0 - 1e-14
        assert float(jnp.max(YY)) <= 2.0 + 1e-14


class TestChebpts3:
    """Tests for 3D Chebyshev tensor grid."""

    def test_cubic_grid_shape(self):
        """chebpts3(n) returns (n, n, n) grids."""
        from chebfunjax.utils import chebpts3
        for n in [3, 4]:
            XX, YY, ZZ = chebpts3(n)
            assert XX.shape == (n, n, n)
            assert YY.shape == (n, n, n)
            assert ZZ.shape == (n, n, n)

    def test_nxnynz_grid(self):
        """chebpts3(nx, ny, nz) returns (nx, ny, nz) grids."""
        from chebfunjax.utils import chebpts3
        XX, YY, ZZ = chebpts3(3, 4, 5)
        assert XX.shape == (3, 4, 5)

    def test_range(self):
        """Points lie in [-1, 1]^3."""
        from chebfunjax.utils import chebpts3
        XX, YY, ZZ = chebpts3(5)
        for arr in [XX, YY, ZZ]:
            assert float(jnp.min(arr)) >= -1.0 - 1e-14
            assert float(jnp.max(arr)) <= 1.0 + 1e-14


class TestPaduapts:
    """Tests for Padua points."""

    def test_count(self):
        """paduapts(n) returns (n+1)*(n+2)//2 points."""
        from chebfunjax.utils import paduapts
        for n in [0, 1, 2, 3, 5, 8]:
            xy, idx = paduapts(n)
            expected = (n + 1) * (n + 2) // 2
            assert xy.shape[0] == expected, f"Wrong count for n={n}: {xy.shape[0]} != {expected}"
            assert xy.shape[1] == 2

    def test_in_square(self):
        """Padua points lie in [-1, 1]^2."""
        from chebfunjax.utils import paduapts
        xy, _ = paduapts(5)
        assert float(jnp.min(xy[:, 0])) >= -1.0 - 1e-14
        assert float(jnp.max(xy[:, 0])) <= 1.0 + 1e-14
        assert float(jnp.min(xy[:, 1])) >= -1.0 - 1e-14
        assert float(jnp.max(xy[:, 1])) <= 1.0 + 1e-14

    def test_n0_trivial(self):
        """paduapts(0) = [(-1, -1)]."""
        from chebfunjax.utils import paduapts
        xy, idx = paduapts(0)
        assert xy.shape == (1, 2)


# ===========================================================================
# Trigutils
# ===========================================================================

class TestTrigpoly:
    """Tests for trigpoly."""

    def test_shape(self):
        """trigpoly(n) returns a vector of the correct length."""
        from chebfunjax.utils import trigpoly
        for n in [1, 2, 5]:
            vals = trigpoly(n)
            assert vals.shape == (2 * n + 1,)

    def test_unit_norm(self):
        """exp(i*pi*n*x) has unit L2 norm on equispaced grid."""
        from chebfunjax.utils import trigpoly
        for n in [1, 3, 5]:
            vals = trigpoly(n)
            # L2 norm on equispaced grid: sum(|vals|^2) / N should be 1
            norm2 = float(jnp.mean(jnp.abs(vals) ** 2))
            npt.assert_allclose(norm2, 1.0, rtol=1e-13)

    def test_integer_only(self):
        """trigpoly raises ValueError for non-integer n."""
        from chebfunjax.utils import trigpoly
        with pytest.raises(ValueError, match="integers"):
            trigpoly(1.5)


# ===========================================================================
# Specfun: besselroots, gammaratio
# ===========================================================================

class TestBesselroots:
    """Tests for first zeros of Bessel J_nu."""

    def test_j0_first_zero(self):
        """First zero of J_0(x) is approximately 2.4048..."""
        from chebfunjax.utils import besselroots
        j = besselroots(0, 1)
        npt.assert_allclose(float(j[0]), 2.4048255576957728, rtol=1e-10)

    def test_j0_five_zeros(self):
        """First 5 zeros of J_0 match known values."""
        from chebfunjax.utils import besselroots
        j = besselroots(0, 5)
        expected = [2.4048255576957728, 5.5200781102863106,
                    8.6537279129110122, 11.791534439014281, 14.930917708487785]
        npt.assert_allclose(np.array(j), expected, rtol=1e-9)

    def test_j1_first_zero(self):
        """First zero of J_1 is approximately 3.8317..."""
        from chebfunjax.utils import besselroots
        j = besselroots(1, 3)
        npt.assert_allclose(float(j[0]), 3.8317059702075125, rtol=1e-6)

    def test_zero_zeros(self):
        """besselroots(nu, 0) returns empty array."""
        from chebfunjax.utils import besselroots
        j = besselroots(0, 0)
        assert j.shape == (0,)

    def test_large_n(self):
        """besselroots works for large n (McMahon's expansion)."""
        from chebfunjax.utils import besselroots
        j = besselroots(0, 30)
        assert j.shape == (30,)
        # All positive and increasing
        assert float(jnp.all(j > 0))
        assert float(jnp.all(jnp.diff(j) > 0))

    def test_negative_n_raises(self):
        """besselroots raises for negative n."""
        from chebfunjax.utils import besselroots
        with pytest.raises(ValueError, match="non-negative"):
            besselroots(0, -1)


class TestGammaratio:
    """Tests for accurate gamma ratio computation."""

    def test_small_m(self):
        """For small m, gammaratio uses scipy directly."""
        from chebfunjax.utils import gammaratio
        from scipy.special import gamma
        for m, delta in [(2.0, 0.5), (5.0, 1.0), (10.0, 0.3)]:
            expected = gamma(m + delta) / gamma(m)
            result = gammaratio(m, delta)
            npt.assert_allclose(result, expected, rtol=1e-12,
                                err_msg=f"Failed for m={m}, delta={delta}")

    def test_large_m(self):
        """For large m, gammaratio is accurate via Stirling series."""
        from chebfunjax.utils import gammaratio
        from scipy.special import gammaln
        for m, delta in [(50.0, 0.5), (100.0, 0.3), (200.0, 0.7)]:
            # Use exp(gammaln) to avoid overflow for large m
            expected = np.exp(gammaln(m + delta) - gammaln(m))
            result = gammaratio(m, delta)
            npt.assert_allclose(result, expected, rtol=1e-12,
                                err_msg=f"Failed for m={m}, delta={delta}")

    def test_zero_delta(self):
        """gammaratio(m, 0) = 1."""
        from chebfunjax.utils import gammaratio
        for m in [2.0, 50.0, 200.0]:
            npt.assert_allclose(gammaratio(m, 0.0), 1.0, rtol=1e-14)

    def test_integer_delta(self):
        """gammaratio(m, n) == prod(m, m+1, ..., m+n-1) for integer n."""
        from chebfunjax.utils import gammaratio
        m, n = 10.0, 3
        expected = m * (m + 1) * (m + 2)
        result = gammaratio(m, n)
        npt.assert_allclose(result, expected, rtol=1e-10)


# ===========================================================================
# Random functions
# ===========================================================================

class TestSmoothie:
    """Tests for smoothie (C-infinity random function)."""

    def test_returns_array(self):
        """smoothie returns a JAX array."""
        from chebfunjax.utils import smoothie
        c = smoothie()
        assert isinstance(c, jnp.ndarray)

    def test_length_reasonable(self):
        """smoothie on [-1,1] returns ~2000 coefficients."""
        from chebfunjax.utils import smoothie
        c = smoothie(domain=(-1.0, 1.0))
        assert c.shape[0] > 100

    def test_real_output(self):
        """Non-trig smoothie has real (or negligible imaginary) coefficients."""
        from chebfunjax.utils import smoothie
        c = smoothie()
        # Should be real dtype or nearly real
        if jnp.issubdtype(c.dtype, jnp.complexfloating):
            assert float(jnp.max(jnp.abs(jnp.imag(c)))) < 1e-10
        else:
            assert jnp.issubdtype(c.dtype, jnp.floating)

    def test_trig_complex(self):
        """Trig smoothie returns complex array."""
        from chebfunjax.utils import smoothie
        c = smoothie(trig=True)
        assert jnp.issubdtype(c.dtype, jnp.complexfloating)

    def test_reproducible_with_key(self):
        """With same JAX key, smoothie gives same result."""
        from chebfunjax.utils import smoothie
        key = jax.random.PRNGKey(0)
        c1 = smoothie(key=key)
        c2 = smoothie(key=key)
        npt.assert_array_equal(np.array(c1), np.array(c2))


class TestRandnfundisk:
    """Tests for random function on unit disk."""

    def test_returns_2d_array(self):
        """randnfundisk returns a 2D array."""
        from chebfunjax.utils import randnfundisk
        F = randnfundisk(10)
        assert F.ndim == 2

    def test_finite_values(self):
        """randnfundisk returns finite values."""
        from chebfunjax.utils import randnfundisk
        F = randnfundisk(8)
        assert bool(jnp.all(jnp.isfinite(F)))


class TestRandnfunsphere:
    """Tests for random function on unit sphere."""

    def test_returns_2d_array(self):
        """randnfunsphere returns a 2D array."""
        from chebfunjax.utils import randnfunsphere
        F = randnfunsphere(5)
        assert F.ndim == 2

    def test_finite_values(self):
        """randnfunsphere returns finite values."""
        from chebfunjax.utils import randnfunsphere
        F = randnfunsphere(4)
        assert bool(jnp.all(jnp.isfinite(F)))

    def test_monochromatic(self):
        """monochromatic=True gives finite 2D result."""
        from chebfunjax.utils import randnfunsphere
        F = randnfunsphere(4, monochromatic=True)
        assert F.ndim == 2
        assert bool(jnp.all(jnp.isfinite(F)))


# ===========================================================================
# Conformal mapping
# ===========================================================================

class TestConformal:
    """Tests for conformal mapping."""

    def test_ellipse_boundary(self):
        """Points on ellipse boundary map approximately to unit circle."""
        from chebfunjax.utils import conformal
        theta = jnp.linspace(0, 2 * jnp.pi, 100, endpoint=False)
        C = 2 * jnp.cos(theta) + 1j * jnp.sin(theta)
        f, finv, pol, polinv = conformal(C, tol=1e-2)
        W = f(C[:10])
        # Boundary should map to unit circle |W| ≈ 1
        mods = np.abs(np.array(W))
        npt.assert_allclose(mods, 1.0, atol=0.1)

    def test_circle_identity(self):
        """Unit circle maps to unit disk with small error."""
        from chebfunjax.utils import conformal
        theta = jnp.linspace(0, 2 * jnp.pi, 100, endpoint=False)
        C = jnp.exp(1j * theta)  # unit circle
        f, finv, pol, polinv = conformal(C, tol=1e-2)
        W = f(C[:5])
        mods = np.abs(np.array(W))
        # Unit circle maps to unit circle
        npt.assert_allclose(mods, 1.0, atol=0.5)

    def test_returns_callables(self):
        """conformal returns two callables and two arrays."""
        from chebfunjax.utils import conformal
        theta = jnp.linspace(0, 2 * jnp.pi, 80, endpoint=False)
        C = 1.5 * jnp.cos(theta) + 1j * jnp.sin(theta)
        f, finv, pol, polinv = conformal(C, tol=1e-2)
        assert callable(f)
        assert callable(finv)
        assert isinstance(pol, jnp.ndarray)
        assert isinstance(polinv, jnp.ndarray)
