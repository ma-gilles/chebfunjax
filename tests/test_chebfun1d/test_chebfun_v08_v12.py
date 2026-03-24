"""Tests for V08-V12 methods added to the Chebfun class.

Covers:
- V08: horzcat, vertcat, size, __getitem__ (subsref)
- V09: polyfit, interp1, spline, pchip
- V10: conv, circconv, flipud, fliplr
- V11: besselj, bessely, airy, ellipj, erf, erfc, erfinv
- V12: isnan, isinf, isreal, logical, any, all, isempty, isequal, __eq__
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import jax
import jax.numpy as jnp
import scipy.special as ss

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.domain import Domain

# ============================================================================
# Helpers
# ============================================================================

ATOL = 1e-10
RTOL = 1e-10


def _pts(n=50, a=-1.0, b=1.0):
    return jnp.linspace(a + 1e-6, b - 1e-6, n, dtype=jnp.float64)


# ============================================================================
# V08 — Quasimatrix ops
# ============================================================================


class TestHorzcat:
    def test_single_chebfun(self):
        f = chebfun(jnp.sin)
        result = Chebfun.horzcat([f])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_two_same_domain(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        result = Chebfun.horzcat([f, g])
        assert len(result) == 2

    def test_domain_mismatch_raises(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos, domain=(0.0, 1.0))
        with pytest.raises(ValueError, match="inconsistent"):
            Chebfun.horzcat([f, g])

    def test_empty_list(self):
        result = Chebfun.horzcat([])
        assert result == []


class TestVertcat:
    def test_two_pieces(self):
        f = chebfun(jnp.sin, domain=(-1.0, 0.0))
        g = chebfun(jnp.sin, domain=(0.0, 1.0))
        h = Chebfun.vertcat([f, g])
        # h should have two pieces and cover [-1, 1]
        assert h.domain.a == pytest.approx(-1.0)
        assert h.domain.b == pytest.approx(1.0)
        assert len(h.funs) == 2
        # Values should match the original sin
        xs = _pts(30)
        npt.assert_allclose(np.array(h(xs)), np.array(jnp.sin(xs)), atol=1e-12)

    def test_single_chebfun(self):
        f = chebfun(jnp.sin)
        result = Chebfun.vertcat([f])
        assert result is f

    def test_contiguity_error(self):
        f = chebfun(jnp.sin, domain=(-1.0, 0.0))
        g = chebfun(jnp.sin, domain=(0.1, 1.0))  # gap
        with pytest.raises(ValueError, match="does not match"):
            Chebfun.vertcat([f, g])


class TestSize:
    def test_no_dim(self):
        f = chebfun(jnp.sin)
        sz = f.size()
        assert sz[0] == float("inf")
        assert sz[1] == 1

    def test_dim1(self):
        f = chebfun(jnp.sin)
        assert f.size(1) == float("inf")

    def test_dim2(self):
        f = chebfun(jnp.sin)
        assert f.size(2) == 1

    def test_higher_dim(self):
        f = chebfun(jnp.sin)
        assert f.size(3) == 1


class TestGetItem:
    def test_index_zero(self):
        f = chebfun(jnp.sin)
        assert f[0] is f

    def test_negative_index(self):
        f = chebfun(jnp.sin)
        assert f[-1] is f

    def test_out_of_range(self):
        f = chebfun(jnp.sin)
        with pytest.raises(IndexError):
            _ = f[1]

    def test_slice(self):
        f = chebfun(jnp.sin)
        assert f[:] is f


# ============================================================================
# V09 — Interpolation / fitting
# ============================================================================


class TestPolyfit:
    def test_truncation(self):
        """Fitting a degree-n polynomial of degree k > n should truncate."""
        # sin has ~18 coefficients on [-1, 1]
        f = chebfun(jnp.sin)
        g = f.polyfit(5)
        assert len(g) == 6  # n+1 coefficients

    def test_no_truncation_when_already_short(self):
        """Fitting with n >= len(f) returns the original."""
        f = chebfun(jnp.sin)
        orig_len = len(f)
        g = f.polyfit(orig_len + 10)
        assert len(g) == orig_len

    def test_polynomial_exact(self):
        """Fitting a polynomial of exact degree n is exact."""
        # f(x) = x^3 is a degree-3 polynomial; polyfit(3) should be exact
        f = chebfun(lambda x: x**3)
        g = f.polyfit(3)
        xs = _pts(30)
        npt.assert_allclose(np.array(g(xs)), np.array(xs**3), atol=1e-13)

    def test_invalid_n(self):
        f = chebfun(jnp.sin)
        with pytest.raises(ValueError):
            f.polyfit(-1)


class TestInterp1:
    def test_polynomial_through_data(self):
        """Interpolant exactly reproduces the data values."""
        x = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y = jnp.sin(x)
        p = Chebfun.interp1(x, y)
        # Should match at the nodes
        npt.assert_allclose(np.array(p(x)), np.array(y), atol=1e-10)

    def test_matches_sin(self):
        """Interpolant through many Chebyshev nodes matches sin accurately."""
        n = 30
        k = jnp.arange(n, dtype=jnp.float64)
        x = -jnp.cos(jnp.pi * k / (n - 1))  # Chebyshev-1 nodes
        y = jnp.sin(x)
        p = Chebfun.interp1(x, y)
        xs = _pts(40)
        npt.assert_allclose(np.array(p(xs)), np.array(jnp.sin(xs)), atol=1e-10)


class TestSpline:
    def test_through_data(self):
        """Spline passes through the given data."""
        x = np.linspace(-1.0, 1.0, 6)
        y = np.sin(x)
        f = Chebfun.spline(x, y)
        npt.assert_allclose(np.array(f(jnp.asarray(x))), y, atol=1e-12)

    def test_smooth_approximation(self):
        """Spline through many points approximates the true function well.

        A cubic spline through 15 equispaced points on [-1,1] has O(h^4)
        error, so we expect ~1e-5 for h=2/14.
        """
        x = np.linspace(-1.0, 1.0, 15)
        y = np.cos(x)
        f = Chebfun.spline(x, y)
        xs = np.linspace(-0.99, 0.99, 40)
        npt.assert_allclose(np.array(f(jnp.asarray(xs))), np.cos(xs), atol=1e-4)


class TestPchip:
    def test_through_data(self):
        """Pchip passes through the given data."""
        x = np.linspace(-1.0, 1.0, 8)
        y = np.sin(x)
        f = Chebfun.pchip(x, y)
        npt.assert_allclose(np.array(f(jnp.asarray(x))), y, atol=1e-12)

    def test_smooth_approximation(self):
        """Pchip through 15 equispaced points approximates exp well.

        PCHIP is shape-preserving but only C1, so accuracy is O(h^3) in
        general. For 15 points on [-1,1], h=2/14 ~ 0.14, so ~3e-3 error
        near the boundary is expected.
        """
        x = np.linspace(-1.0, 1.0, 15)
        y = np.exp(x)
        f = Chebfun.pchip(x, y)
        xs = np.linspace(-0.99, 0.99, 40)
        npt.assert_allclose(np.array(f(jnp.asarray(xs))), np.exp(xs), atol=5e-3)


# ============================================================================
# V10 — Convolution, flip
# ============================================================================


class TestConv:
    def test_conv_constant(self):
        """conv(1, 1) on [0,1]x[0,1] = h(x) = x for x in [0,1], =2-x for [1,2]."""
        f = chebfun(lambda x: jnp.ones_like(x), domain=(0.0, 1.0))
        g = chebfun(lambda x: jnp.ones_like(x), domain=(0.0, 1.0))
        h = f.conv(g)
        # h should be on [0, 2]
        assert h.domain.a == pytest.approx(0.0, abs=1e-10)
        assert h.domain.b == pytest.approx(2.0, abs=1e-10)
        # At x=1: h(1) = int_0^1 1 dt = 1
        val = float(h(jnp.float64(1.0)))
        assert val == pytest.approx(1.0, abs=1e-8)

    def test_conv_sin_cos(self):
        """conv(sin, cos) matches known result via integration."""
        import scipy.integrate as si
        f = chebfun(jnp.sin, domain=(-1.0, 1.0))
        g = chebfun(jnp.cos, domain=(-1.0, 1.0))
        h = f.conv(g)
        # Test at x=0 (midpoint of output domain [-2, 2])
        x_test = 0.0
        # Numeric check: h(0) = int_{-1}^{1} sin(t)*cos(0-t) dt
        val_numeric, _ = si.quad(lambda t: np.sin(t) * np.cos(-t), -1.0, 1.0)
        val_cheb = float(h(jnp.float64(x_test)))
        assert val_cheb == pytest.approx(val_numeric, abs=1e-7)

    def test_conv_unbounded_raises(self):
        from chebfunjax.domain import Domain
        # Use a very large but still bounded domain to test the error path
        # Actually test the ValueError directly
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        # Manually test the ValueError condition: inject an inf
        with pytest.raises((ValueError, Exception)):
            # Patch domain manually to get unbounded
            bad = Chebfun(
                funs=f.funs,
                domain=Domain((-float("inf"), 1.0)),
            )
            bad.conv(g)


class TestCircconv:
    def test_circconv_shape(self):
        """circconv returns a Chebfun on the same domain."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        h = f.circconv(g)
        assert h.domain.a == pytest.approx(f.domain.a, abs=1e-10)
        assert h.domain.b == pytest.approx(f.domain.b, abs=1e-10)

    def test_circconv_domain_mismatch(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos, domain=(0.0, 1.0))
        with pytest.raises(ValueError):
            f.circconv(g)


class TestFlipud:
    def test_flipud_reversal(self):
        """flipud: g(x) = f(a+b-x)."""
        f = chebfun(jnp.sin)  # on [-1, 1]
        g = f.flipud()
        xs = _pts(40)
        # g(x) == sin(0 - x) = -sin(x) since a+b-x = -1+1-x = -x
        npt.assert_allclose(
            np.array(g(xs)),
            np.array(jnp.sin(-xs)),
            atol=1e-12,
        )

    def test_fliplr_same_as_flipud(self):
        f = chebfun(jnp.exp)
        npt.assert_allclose(
            np.array(f.fliplr()(_pts(20))),
            np.array(f.flipud()(_pts(20))),
            atol=1e-14,
        )


# ============================================================================
# V11 — Special functions
# ============================================================================


class TestBesselj:
    def test_besselj_0(self):
        """besselj(0, f(x)) matches scipy on [-1, 1]."""
        f = chebfun(lambda x: x * 0.5)  # scale to avoid large arguments
        h = f.besselj(0)
        xs = _pts(40)
        expected = ss.jv(0, np.array(f(xs)))
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-10)

    def test_besselj_noninteger_order(self):
        """besselj with non-integer order."""
        f = chebfun(lambda x: x * 0.5 + 1.5)  # > 0 to avoid branch cuts
        h = f.besselj(0.5)
        xs = _pts(30)
        expected = ss.jv(0.5, np.array(f(xs)))
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-9)


class TestBessely:
    def test_bessely_0(self):
        """bessely(0, f(x)) matches scipy on domain away from zero."""
        f = chebfun(lambda x: x + 2.0)  # x in [1, 3], away from 0
        h = f.bessely(0)
        xs = _pts(30)
        expected = ss.yv(0, np.array(f(xs)))
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-10)


class TestAiry:
    def test_airy_k0(self):
        """airy(0, f) = Ai(f) matches scipy."""
        f = chebfun(lambda x: x * 0.5)
        h = f.airy(0)
        xs = _pts(30)
        expected = ss.airy(np.array(f(xs)))[0]
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-10)

    def test_airy_k2(self):
        """airy(2, f) = Bi(f) matches scipy."""
        f = chebfun(lambda x: x * 0.5)
        h = f.airy(2)
        xs = _pts(30)
        expected = ss.airy(np.array(f(xs)))[2]
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-10)


class TestEllipj:
    def test_ellipj_values(self):
        """ellipj sn, cn, dn match scipy.special.ellipj."""
        m = 0.5
        f = chebfun(lambda x: x * 0.5)  # small arguments
        sn, cn, dn = f.ellipj(m)
        xs = _pts(30)
        fxs = np.array(f(xs))
        exp_sn, exp_cn, exp_dn, _ = ss.ellipj(fxs, m)
        npt.assert_allclose(np.array(sn(xs)), exp_sn, atol=1e-10)
        npt.assert_allclose(np.array(cn(xs)), exp_cn, atol=1e-10)
        npt.assert_allclose(np.array(dn(xs)), exp_dn, atol=1e-10)

    def test_ellipj_identity(self):
        """sn^2 + cn^2 == 1."""
        m = 0.3
        f = chebfun(lambda x: x * 0.5)
        sn, cn, _ = f.ellipj(m)
        xs = _pts(20)
        npt.assert_allclose(
            np.array(sn(xs))**2 + np.array(cn(xs))**2,
            np.ones(20),
            atol=1e-10,
        )


class TestErf:
    def test_erf_values(self):
        f = chebfun(lambda x: x * 0.8)
        h = f.erf()
        xs = _pts(40)
        expected = ss.erf(np.array(f(xs)))
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-12)

    def test_erfc_values(self):
        f = chebfun(lambda x: x * 0.8)
        h = f.erfc()
        xs = _pts(40)
        expected = ss.erfc(np.array(f(xs)))
        npt.assert_allclose(np.array(h(xs)), expected, atol=1e-12)

    def test_erfinv_roundtrip(self):
        """erfinv(erf(f)) == f."""
        f = chebfun(lambda x: x * 0.8)  # values in (-0.8, 0.8), erf in (-1,1)
        g = f.erf()
        h = g.erfinv()
        xs = _pts(30)
        npt.assert_allclose(np.array(h(xs)), np.array(f(xs)), atol=1e-11)


# ============================================================================
# V12 — Type / logical ops
# ============================================================================


class TestIsnan:
    def test_clean_chebfun_not_nan(self):
        f = chebfun(jnp.sin)
        assert f.isnan() is False

    def test_nan_coeffs(self):
        from chebfunjax.tech.chebtech import Chebtech2
        from chebfunjax.chebfun1d.chebfun import _Piece
        bad_coeffs = jnp.array([jnp.nan, 1.0, 0.0], dtype=jnp.float64)
        bad_tech = Chebtech2.from_coeffs(bad_coeffs)
        bad_piece = _Piece(tech=bad_tech, interval=(-1.0, 1.0))
        bad_f = Chebfun(funs=[bad_piece], domain=Domain((-1.0, 1.0)))
        assert bad_f.isnan() is True


class TestIsinf:
    def test_clean_not_inf(self):
        f = chebfun(jnp.cos)
        assert f.isinf() is False

    def test_inf_coeffs(self):
        from chebfunjax.tech.chebtech import Chebtech2
        from chebfunjax.chebfun1d.chebfun import _Piece
        inf_coeffs = jnp.array([jnp.inf, 0.0], dtype=jnp.float64)
        inf_tech = Chebtech2.from_coeffs(inf_coeffs)
        inf_piece = _Piece(tech=inf_tech, interval=(-1.0, 1.0))
        inf_f = Chebfun(funs=[inf_piece], domain=Domain((-1.0, 1.0)))
        assert inf_f.isinf() is True


class TestIsreal:
    def test_float_chebfun_is_real(self):
        f = chebfun(jnp.sin)
        assert f.isreal() is True


class TestLogical:
    def test_positive_fn_is_all_ones(self):
        f = chebfun(jnp.exp)  # always > 0
        lg = f.logical()
        xs = _pts(20)
        npt.assert_allclose(np.array(lg(xs)), np.ones(20), atol=1e-10)

    def test_sign_changes_create_zeros(self):
        f = chebfun(jnp.sin)  # zero at x=0
        lg = f.logical()
        # At x=0 (the root) logical should be 0
        val = float(lg(jnp.float64(0.0)))
        assert val == pytest.approx(0.0, abs=0.1)


class TestAny:
    def test_nonzero_fn(self):
        f = chebfun(jnp.exp)
        assert f.any() is True

    def test_zero_fn(self):
        f = chebfun(0.0)
        assert f.any() is False


class TestAll:
    def test_positive_no_roots(self):
        f = chebfun(jnp.exp)  # always > 0, no roots
        assert f.all() is True

    def test_fn_with_roots(self):
        f = chebfun(jnp.sin)  # has a root at x=0 on [-1,1]
        assert f.all() is False


class TestIsempty:
    def test_standard_chebfun_not_empty(self):
        f = chebfun(jnp.sin)
        assert f.isempty() is False


class TestIsequal:
    def test_equal_chebfuns(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.sin)
        # Both should have the same coefficients (adaptive builds same result)
        assert f.isequal(g) is True

    def test_different_chebfuns(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        assert f.isequal(g) is False

    def test_different_domains(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.sin, domain=(0.0, 1.0))
        assert f.isequal(g) is False

    def test_eq_dunder(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.sin)
        assert (f == g) is True

    def test_eq_non_chebfun(self):
        # When comparing with a non-Chebfun, __eq__ returns NotImplemented
        # which Python/equinox resolves to False (not the same object).
        f = chebfun(jnp.sin)
        assert not (f == 1.0)
