"""Core-suite coverage mirrors for :mod:`chebfunjax.fun.singfun`.

These tests exercise the recently-added Singfun methods and private helpers
(exponent canonicalisation, complex parts, division that creates poles,
Case-3 addition, restrict endpoint pieces, chebcoeffs, singular cumsum, and
the automatic exponent finders) that the MATLAB-port suite covers but the
core suite did not.  Every assertion checks a closed-form value, not merely
that the code runs.

Provenance
----------
Mirrors of MATLAB @singfun tests; Chebfun commit 7574c77.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.fun.singfun import (
    Singfun,
    _chebT2U,
    _jacobi_moments,
)
from chebfunjax.tech.chebtech import Chebtech2

RTOL = 1e-10


def _teval(coeffs, x):
    """Evaluate a first-kind Chebyshev T-series at scalar x."""
    c = np.asarray(coeffs)
    k = np.arange(c.shape[0])
    return float(np.sum(c * np.cos(k * np.arccos(x))))


# ----------------------------------------------------------------------
# Division: non-pole branch, pole-creating branch, reciprocal
# ----------------------------------------------------------------------
class TestSingfunDivision:
    def test_div_nonvanishing_smoothpart(self):
        # f = 2 (1+x)^0.5, g = (1+x)^0.2 ; both smooth parts nonzero at ends.
        fs = Singfun(2.0, (0.5, 0.0))
        gs = Singfun(1.0, (0.2, 0.0))
        q = fs / gs
        for x in (-0.3, 0.0, 0.5, 0.9):
            npt.assert_allclose(float(q(jnp.float64(x))), 2.0 * (1.0 + x) ** 0.3, rtol=RTOL)

    def test_div_creates_pole(self):
        # g smooth part vanishes at the left end -> quotient has a pole there.
        fs = Singfun(1.0, (0.0, 0.0))
        gs = Singfun(Chebtech2.from_function(lambda x: 1.0 + x), (0.0, 0.0))
        q = fs / gs
        for x in (-0.5, 0.0, 0.5, 0.9):
            npt.assert_allclose(float(q(jnp.float64(x))), 1.0 / (1.0 + x), rtol=1e-8)

    def test_scalar_over_singfun(self):
        # 2 / ((1+x)^0.5) = 2 (1+x)^-0.5
        fs = Singfun(1.0, (0.5, 0.0))
        q = 2.0 / fs
        for x in (-0.5, 0.0, 0.5, 0.9):
            npt.assert_allclose(float(q(jnp.float64(x))), 2.0 * (1.0 + x) ** -0.5, rtol=RTOL)

    def test_div_by_scalar(self):
        fs = Singfun(4.0, (0.5, 0.0))
        q = fs / 2.0
        npt.assert_allclose(float(q(jnp.float64(0.5))), 2.0 * (1.5) ** 0.5, rtol=RTOL)


# ----------------------------------------------------------------------
# Addition Case 3 (non-integer exponent difference)
# ----------------------------------------------------------------------
class TestSingfunAddCase3:
    def test_add_noninteger_diff(self):
        f = Singfun(1.0, (0.5, 0.0))
        g = Singfun(1.0, (0.3, 0.0))
        s = f + g
        # smaller exponent wins as the common factor
        npt.assert_allclose(s.exponents[0], 0.3, atol=1e-12)
        for x in (-0.2, 0.0, 0.4, 0.8):
            expect = (1.0 + x) ** 0.5 + (1.0 + x) ** 0.3
            npt.assert_allclose(float(s(jnp.float64(x))), expect, rtol=1e-6)

    def test_sub_noninteger_diff(self):
        f = Singfun(1.0, (0.0, 0.6))
        g = Singfun(1.0, (0.0, 0.25))
        s = f - g
        for x in (-0.8, 0.4):
            expect = (1.0 - x) ** 0.6 - (1.0 - x) ** 0.25
            npt.assert_allclose(float(s(jnp.float64(x))), expect, rtol=1e-6, atol=1e-7)


# ----------------------------------------------------------------------
# Boundary-root extraction and exponent canonicalisation
# ----------------------------------------------------------------------
class TestSingfunExponentCanon:
    def test_extract_boundary_roots_auto(self):
        # smooth part has a simple root at x = -1
        sp = Chebtech2.from_function(lambda x: (1.0 + x) * (2.0 + x))
        f = Singfun(sp, (0.0, 0.0))
        g = f.extractBoundaryRoots()
        npt.assert_allclose(g.exponents[0], 1.0, atol=1e-9)
        # function value is unchanged in the interior
        for x in (-0.5, 0.0, 0.7):
            npt.assert_allclose(float(g(jnp.float64(x))), float(f(jnp.float64(x))), rtol=1e-9)

    def test_extract_boundary_roots_explicit(self):
        sp = Chebtech2.from_function(lambda x: (1.0 - x) * (3.0 - x))
        f = Singfun(sp, (0.0, 0.0))
        g = f.extractBoundaryRoots((0.0, 1.0))
        npt.assert_allclose(g.exponents[1], 1.0, atol=1e-9)

    def test_cancel_exponents(self):
        # (1+x) / (1+x) = 1 : negative exponent cancels the boundary root
        sp = Chebtech2.from_function(lambda x: 1.0 + x)
        f = Singfun(sp, (-1.0, 0.0))
        g = f.cancelExponents()
        npt.assert_allclose(g.exponents[0], 0.0, atol=1e-9)
        for x in (-0.5, 0.0, 0.6):
            npt.assert_allclose(float(g(jnp.float64(x))), 1.0, rtol=1e-8)

    def test_cancel_exponents_noop(self):
        # negative exponent but smooth part does NOT vanish -> unchanged
        f = Singfun(1.0, (-0.5, 0.0))
        g = f.cancelExponents()
        npt.assert_allclose(g.exponents[0], -0.5, atol=1e-12)

    def test_simplify_exponents(self):
        # (1+x)^1.5 -> smooth part (1+x) with exponent 0.5
        f = Singfun(1.0, (1.5, 0.0))
        g = f.simplifyExponents()
        npt.assert_allclose(g.exponents[0], 0.5, atol=1e-12)
        npt.assert_allclose(float(g(jnp.float64(0.5))), 1.5 ** 1.5, rtol=RTOL)

    def test_simplify_exponents_below_one_noop(self):
        f = Singfun(1.0, (0.4, 0.2))
        g = f.simplifyExponents()
        npt.assert_allclose(g.exponents, (0.4, 0.2), atol=1e-12)

    def test_simplify_with_and_without_tol(self):
        f = Singfun(1.0, (1.25, 0.0))
        g = f.simplify()
        npt.assert_allclose(g.exponents[0], 0.25, atol=1e-12)
        npt.assert_allclose(float(g(jnp.float64(0.3))), 1.3 ** 1.25, rtol=RTOL)
        g2 = f.simplify(1e-13)
        npt.assert_allclose(float(g2(jnp.float64(0.3))), 1.3 ** 1.25, rtol=RTOL)


# ----------------------------------------------------------------------
# Complex parts on a smooth (issmooth) Singfun
# ----------------------------------------------------------------------
class TestSingfunComplexParts:
    def _cplx(self):
        sp = Chebtech2.from_function(lambda x: x + 1j * x ** 2)
        return Singfun(sp, (0.0, 0.0))

    def test_real_of_smooth_complex(self):
        f = self._cplx()
        r = f.real()
        npt.assert_allclose(float(r(jnp.float64(0.5)).real), 0.5, rtol=1e-9)

    def test_imag_of_smooth_complex(self):
        f = self._cplx()
        im = f.imag()
        npt.assert_allclose(float(im(jnp.float64(0.5)).real), 0.25, rtol=1e-9)

    def test_conj_of_smooth_complex(self):
        f = self._cplx()
        c = f.conj()
        val = complex(c(jnp.float64(0.5)))
        npt.assert_allclose(val.real, 0.5, rtol=1e-9)
        npt.assert_allclose(val.imag, -0.25, rtol=1e-9)

    def test_real_of_smooth_real_returns_smoothpart(self):
        f = Singfun(Chebtech2.from_function(lambda x: x ** 2), (0.0, 0.0))
        r = f.real()
        assert isinstance(r, Chebtech2)

    def test_complex_parts_singular(self):
        # non-smooth branch keeps the Singfun wrapper
        sp = Chebtech2.from_function(lambda x: x + 1j * x)
        f = Singfun(sp, (0.5, 0.0))
        assert isinstance(f.real(), Singfun)
        assert isinstance(f.imag(), Singfun)
        assert isinstance(f.conj(), Singfun)


# ----------------------------------------------------------------------
# compose with a Chebtech2 / Singfun outer operator
# ----------------------------------------------------------------------
class TestSingfunCompose:
    def test_compose_with_chebtech_outer(self):
        f = Singfun(1.0, (0.5, 0.0))  # (1+x)^0.5
        outer = Chebtech2.from_function(lambda x: x ** 2)
        g = f.compose(outer)  # (f(x))^2 = 1+x
        for x in (-0.5, 0.0, 0.7):
            npt.assert_allclose(float(g(jnp.float64(x))), 1.0 + x, rtol=1e-8)

    def test_compose_with_callable(self):
        f = Singfun(1.0, (0.0, 0.0))
        g = f.compose(lambda v: 2.0 * v + 1.0)
        npt.assert_allclose(float(g(jnp.float64(0.3))), 3.0, rtol=1e-9)

    def test_compose_binary(self):
        f = Singfun(Chebtech2.from_function(lambda x: x), (0.0, 0.0))
        g = Singfun(Chebtech2.from_function(lambda x: x ** 2), (0.0, 0.0))
        h = f.compose(lambda u, v: u + v, g)
        npt.assert_allclose(float(h(jnp.float64(0.4))), 0.4 + 0.16, rtol=1e-8)


# ----------------------------------------------------------------------
# roots with endpoint exponents
# ----------------------------------------------------------------------
class TestSingfunRootsEndpoints:
    def test_left_exponent_adds_root(self):
        f = Singfun(Chebtech2.from_function(lambda x: x), (0.5, 0.0))
        r = np.sort(np.asarray(f.roots()))
        npt.assert_allclose(r, [-1.0, 0.0], atol=1e-9)

    def test_left_exponent_collapses_existing_root(self):
        f = Singfun(Chebtech2.from_function(lambda x: 1.0 + x), (0.5, 0.0))
        r = np.asarray(f.roots())
        npt.assert_allclose(np.sort(r), [-1.0], atol=1e-8)

    def test_left_exponent_no_interior_roots(self):
        f = Singfun(1.0, (0.5, 0.0))
        npt.assert_allclose(np.asarray(f.roots()), [-1.0], atol=1e-12)

    def test_right_exponent_adds_root(self):
        f = Singfun(1.0, (0.0, 0.5))
        npt.assert_allclose(np.asarray(f.roots()), [1.0], atol=1e-12)

    def test_right_exponent_collapses_existing_root(self):
        f = Singfun(Chebtech2.from_function(lambda x: 1.0 - x), (0.0, 0.5))
        r = np.asarray(f.roots())
        npt.assert_allclose(np.sort(r), [1.0], atol=1e-8)


# ----------------------------------------------------------------------
# minandmax: smooth and singular branches
# ----------------------------------------------------------------------
class TestSingfunMinAndMax:
    def test_smooth_branch(self):
        f = Singfun(Chebtech2.from_function(lambda x: x ** 2 - 0.5), (0.0, 0.0))
        vals, pos = f.minandmax()
        npt.assert_allclose(float(vals[0]), -0.5, atol=1e-9)
        npt.assert_allclose(float(vals[1]), 0.5, atol=1e-9)

    def test_left_pole_plus_infinity(self):
        f = Singfun(1.0, (-0.5, 0.0))  # +inf at x = -1
        vals, pos = f.minandmax()
        assert math.isinf(float(vals[1])) and float(vals[1]) > 0
        npt.assert_allclose(float(pos[1]), -1.0, atol=1e-12)
        # finite min at the right endpoint
        npt.assert_allclose(float(vals[0]), 2.0 ** -0.5, rtol=1e-6)

    def test_left_pole_minus_infinity(self):
        f = Singfun(-1.0, (-0.5, 0.0))  # -inf at x = -1
        vals, pos = f.minandmax()
        assert math.isinf(float(vals[0])) and float(vals[0]) < 0
        npt.assert_allclose(float(pos[0]), -1.0, atol=1e-12)

    def test_right_pole_plus_infinity(self):
        f = Singfun(1.0, (0.0, -0.5))  # +inf at x = 1
        vals, pos = f.minandmax()
        assert math.isinf(float(vals[1])) and float(vals[1]) > 0
        npt.assert_allclose(float(pos[1]), 1.0, atol=1e-12)


# ----------------------------------------------------------------------
# restrict: endpoint pieces with both exponents nonzero
# ----------------------------------------------------------------------
class TestSingfunRestrictPieces:
    def test_invalid_interval_raises(self):
        f = Singfun(1.0, (0.5, 0.0))
        with pytest.raises(ValueError):
            f.restrict([0.5, 0.0])

    def test_whole_interval_returns_self(self):
        f = Singfun(1.0, (0.5, 0.3))
        assert f.restrict([-1.0, 1.0]) is f

    def test_left_piece_both_exponents(self):
        f = Singfun(1.0, (0.5, 0.3))  # (1+x)^0.5 (1-x)^0.3
        piece = f.restrict([-1.0, 0.0])
        assert isinstance(piece, Singfun)
        # midpoint t = 0 maps to x = -0.5
        npt.assert_allclose(
            float(piece(jnp.float64(0.0))), float(f(jnp.float64(-0.5))), rtol=1e-8
        )

    def test_right_piece_both_exponents(self):
        f = Singfun(1.0, (0.5, 0.3))
        piece = f.restrict([0.0, 1.0])
        assert isinstance(piece, Singfun)
        # midpoint t = 0 maps to x = 0.5
        npt.assert_allclose(
            float(piece(jnp.float64(0.0))), float(f(jnp.float64(0.5))), rtol=1e-8
        )

    def test_interior_piece(self):
        f = Singfun(Chebtech2.from_function(lambda x: x ** 2), (0.0, 0.0))
        piece = f.restrict([-0.5, 0.5])
        assert isinstance(piece, Chebtech2)
        npt.assert_allclose(float(piece(jnp.float64(0.0))), 0.0, atol=1e-9)

    def test_three_subintervals(self):
        f = Singfun(1.0, (0.5, 0.0))
        pieces = f.restrict([-1.0, 0.0, 1.0])
        assert isinstance(pieces, list) and len(pieces) == 2


# ----------------------------------------------------------------------
# chebcoeffs (kinds 1, 2) and helper _chebT2U
# ----------------------------------------------------------------------
class TestSingfunChebcoeffs:
    def test_kind1_reconstructs_function(self):
        # f = sqrt(1 - x^2)
        f = Singfun(1.0, (0.5, 0.5))
        c = np.asarray(f.chebcoeffs(200, kind=1))
        for x in (-0.6, -0.1, 0.3, 0.7):
            npt.assert_allclose(_teval(c, x), math.sqrt(1.0 - x * x), atol=1e-4)

    def test_kind2_shape_and_finite(self):
        f = Singfun(1.0, (0.5, 0.5))
        c = np.asarray(f.chebcoeffs(6, kind=2))
        assert c.shape == (6,)
        assert np.all(np.isfinite(c))

    def test_bad_kind_raises(self):
        f = Singfun(1.0, (0.5, 0.5))
        with pytest.raises(ValueError):
            f.chebcoeffs(5, kind=3)

    def test_undefined_expansion_raises(self):
        f = Singfun(1.0, (-0.7, 0.0))
        with pytest.raises(ValueError):
            f.chebcoeffs(5)

    def test_chebT2U_identity_on_constant(self):
        cU = np.asarray(_chebT2U(jnp.asarray([1.0], dtype=jnp.float64)))
        npt.assert_allclose(cU, [1.0], atol=1e-12)


# ----------------------------------------------------------------------
# sum: divergent and Gegenbauer branches
# ----------------------------------------------------------------------
class TestSingfunSumBranches:
    def test_both_ends_diverge_opposite_sign_nan(self):
        f = Singfun(Chebtech2.from_function(lambda x: x), (-1.5, -1.5))
        assert math.isnan(float(f.sum()))

    def test_both_ends_diverge_same_sign_inf(self):
        f = Singfun(1.0, (-1.5, -1.5))
        v = float(f.sum())
        assert math.isinf(v) and v > 0

    def test_gegenbauer_even_moments(self):
        # integral of (1 + x^2) sqrt(1 - x^2) over [-1, 1] = 5 pi / 8
        f = Singfun(Chebtech2.from_function(lambda x: 1.0 + x ** 2), (0.5, 0.5))
        npt.assert_allclose(float(f.sum()), 5.0 * math.pi / 8.0, rtol=1e-9)

    def test_left_divergent_inf(self):
        f = Singfun(1.0, (-1.0, 0.0))
        v = float(f.sum())
        assert math.isinf(v) and v > 0


# ----------------------------------------------------------------------
# inner: <f, g> = int conj(f) g dx (conjugate-linear in f, MATLAB
# @singfun/innerProduct.m); singular pairing, complex conjugation,
# smooth-partner promotion, and the innerProduct alias.
# ----------------------------------------------------------------------
class TestSingfunInner:
    def test_real_singular_pairing(self):
        # <sqrt(1-x^2), 1> = integral sqrt(1-x^2) dx = pi/2
        f = Singfun(1.0, (0.5, 0.5))
        g = Singfun(1.0, (0.0, 0.0))
        npt.assert_allclose(float(f.inner(g).real), math.pi / 2.0, rtol=1e-9)

    def test_conjugate_linear_in_f(self):
        # f = 1 + i x, g = x  ->  <f,g> = int conj(1+ix) x dx = -2i/3.
        # The conjugation is essential: without it the result would be +2i/3.
        f = Singfun(Chebtech2.from_function(lambda x: 1.0 + 1j * x), (0.0, 0.0))
        g = Singfun(Chebtech2.from_function(lambda x: x), (0.0, 0.0))
        npt.assert_allclose(complex(f.inner(g)), -2j / 3.0, atol=1e-12)

    def test_promotes_smooth_partner(self):
        # A bare Chebtech2 partner is promoted to a zero-exponent Singfun.
        f = Singfun(1.0, (0.5, 0.5))
        g = Chebtech2.from_function(lambda x: 1.0 + 0.0 * x)
        npt.assert_allclose(float(f.inner(g).real), math.pi / 2.0, rtol=1e-9)

    def test_innerProduct_alias(self):
        f = Singfun(1.0, (0.5, 0.5))
        g = Singfun(1.0, (0.0, 0.0))
        assert complex(f.innerProduct(g)) == complex(f.inner(g))


# ----------------------------------------------------------------------
# cumsum: smooth, both-singular error, one-sided (both orientations)
# ----------------------------------------------------------------------
class TestSingfunCumsum:
    def test_smooth_cumsum(self):
        f = Singfun(Chebtech2.from_function(lambda x: x), (0.0, 0.0))
        F = f.cumsum()
        # antiderivative of x with F(-1)=0 is (x^2 - 1)/2
        npt.assert_allclose(float(F(jnp.float64(0.0))), -0.5, atol=1e-9)
        npt.assert_allclose(float(F(jnp.float64(1.0))), 0.0, atol=1e-9)

    def test_both_singular_not_implemented(self):
        f = Singfun(1.0, (0.5, 0.5))
        with pytest.raises(NotImplementedError):
            f.cumsum()

    def test_left_singularity_cumsum(self):
        # f = (1+x)^-0.5 ; antiderivative 2 (1+x)^0.5, F(-1)=0
        f = Singfun(1.0, (-0.5, 0.0))
        F = f.cumsum()
        npt.assert_allclose(float(F(jnp.float64(0.0))), 2.0, rtol=1e-6)
        npt.assert_allclose(float(F(jnp.float64(1.0))), 2.0 * math.sqrt(2.0), rtol=1e-6)

    def test_right_singularity_cumsum(self):
        # f = (1-x)^-0.5 ; antiderivative 2 sqrt(2) - 2 (1-x)^0.5
        f = Singfun(1.0, (0.0, -0.5))
        F = f.cumsum()
        npt.assert_allclose(float(F(jnp.float64(1.0))), 2.0 * math.sqrt(2.0), rtol=1e-6)
        npt.assert_allclose(
            float(F(jnp.float64(0.0))), 2.0 * math.sqrt(2.0) - 2.0, rtol=1e-6
        )


# ----------------------------------------------------------------------
# Automatic exponent detection (findSingExponents / findPoleOrder / findSingOrder)
# ----------------------------------------------------------------------
class TestSingfunAutoExponents:
    def test_detect_left_pole(self):
        f = Singfun.from_function(lambda x: 1.0 / (1.0 + x))
        npt.assert_allclose(f.exponents[0], -1.0, atol=1e-3)
        npt.assert_allclose(float(f(jnp.float64(0.5))), 1.0 / 1.5, rtol=1e-6)

    def test_detect_left_fractional(self):
        f = Singfun.from_function(lambda x: (1.0 + x) ** 0.5)
        npt.assert_allclose(f.exponents[0], 0.5, atol=1e-2)
        npt.assert_allclose(float(f(jnp.float64(0.3))), 1.3 ** 0.5, rtol=1e-6)

    def test_detect_right_pole(self):
        f = Singfun.from_function(lambda x: 1.0 / (1.0 - x))
        npt.assert_allclose(f.exponents[1], -1.0, atol=1e-3)
        npt.assert_allclose(float(f(jnp.float64(-0.5))), 1.0 / 1.5, rtol=1e-6)


# ----------------------------------------------------------------------
# _jacobi_moments direct sanity (M_0 for the Gegenbauer branch)
# ----------------------------------------------------------------------
class TestJacobiMoments:
    def test_gegenbauer_m0(self):
        # M_0 = int (1-x^2)^0.5 dx = pi/2 for a = b = 0.5
        M = np.asarray(_jacobi_moments(0.5, 0.5, 5))
        npt.assert_allclose(M[0], math.pi / 2.0, rtol=1e-12)
        # odd moments vanish
        npt.assert_allclose(M[1], 0.0, atol=1e-12)

    def test_general_m0(self):
        # M_0 = 2^(a+b+1) B(a+1, b+1) for a != b
        a, b = 0.5, 0.25
        M = np.asarray(_jacobi_moments(a, b, 4))
        expect = (2.0 ** (a + b + 1.0)) * (
            math.gamma(a + 1.0) * math.gamma(b + 1.0) / math.gamma(a + b + 2.0)
        )
        npt.assert_allclose(M[0], expect, rtol=1e-12)
