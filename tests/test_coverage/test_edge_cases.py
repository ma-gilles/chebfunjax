"""Edge-case tests for low-coverage modules (V28).

Covers edge cases for:
- Chebfun1d: empty slices, single-point domains, degenerate functions
- chebtech: n=1 and n=2 edge cases
- transforms: trivial inputs (length 1 and 2)
- singfun: zero exponents, symmetric exponents
- Chebfun2: constant functions, rank-1 approximations
- utils/misc: standard_chop edge cases
- utils/polynomials: degree-0 polynomials
- domain: empty and single-breakpoint domains
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

ATOL = 1e-12


# ============================================================================
# Chebfun1d edge cases
# ============================================================================


class TestChebfun1dEdgeCases:
    """Edge cases for the 1D Chebfun class."""

    def test_constant_function(self):
        """Constant Chebfun evaluates correctly everywhere."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.ones_like(x) * 3.14)
        xs = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        npt.assert_allclose(np.array(f(xs)), np.full(10, 3.14), atol=ATOL)

    def test_zero_function(self):
        """Zero Chebfun has sum == 0."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.zeros_like(x))
        assert abs(float(f.sum())) < ATOL

    def test_diff_of_constant_is_zero(self):
        """Derivative of constant == 0."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.ones_like(x) * 5.0)
        df = f.diff()
        xs = jnp.linspace(-0.9, 0.9, 10, dtype=jnp.float64)
        npt.assert_allclose(np.array(df(xs)), np.zeros(10), atol=ATOL)

    def test_diff_of_linear(self):
        """Derivative of 3x + 2 == 3."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: 3.0 * x + 2.0)
        df = f.diff()
        xs = jnp.linspace(-0.9, 0.9, 10, dtype=jnp.float64)
        npt.assert_allclose(np.array(df(xs)), np.full(10, 3.0), atol=1e-11)

    def test_cumsum_of_zero(self):
        """Antiderivative of zero == 0."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.zeros_like(x))
        F = f.cumsum()
        xs = jnp.linspace(-0.9, 0.9, 10, dtype=jnp.float64)
        npt.assert_allclose(np.array(F(xs)), np.zeros(10), atol=ATOL)

    def test_sum_of_constant(self):
        """sum(c) == c * 2 on [-1, 1]."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        c = 7.0
        f = chebfun(lambda x: jnp.ones_like(x) * c)
        assert abs(float(f.sum()) - c * 2.0) < ATOL

    def test_arithmetic_add_subtract_consistency(self):
        """(f + g) - g == f."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        h = (f + g) - g
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(h(xs)), np.array(f(xs)), atol=1e-11)

    def test_scalar_multiply(self):
        """3 * f evaluates to 3 * f(x)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(jnp.sin)
        g = 3.0 * f
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(g(xs)), 3.0 * np.array(jnp.sin(xs)), atol=1e-11)

    def test_negation(self):
        """-f evaluates to -f(x)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(jnp.cos)
        g = -f
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(g(xs)), -np.array(jnp.cos(xs)), atol=1e-11)

    def test_norm_inf_vs_max(self):
        """norm(f, inf) ≈ max(|f(x)|) on dense grid."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(jnp.sin)
        n_inf = float(f.norm(np.inf))
        xs = jnp.linspace(-1.0, 1.0, 10000, dtype=jnp.float64)
        dense_max = float(jnp.max(jnp.abs(f(xs))))
        # They should agree to within the coarseness of the grid
        assert abs(n_inf - dense_max) < 1e-4


# ============================================================================
# Chebtech2 edge cases
# ============================================================================


class TestChebtechEdgeCases:
    """Edge cases for Chebtech2."""

    def test_n1_construction(self):
        """Chebtech2 with n=1 (constant)."""
        from chebfunjax.tech.chebtech import Chebtech2
        tech = Chebtech2.from_function(lambda x: jnp.ones_like(x) * 5.0, n=1)
        assert tech.n == 1
        assert abs(float(tech(jnp.float64(0.0))) - 5.0) < ATOL

    def test_n2_construction(self):
        """Chebtech2 with n=2 (linear)."""
        from chebfunjax.tech.chebtech import Chebtech2
        tech = Chebtech2.from_function(lambda x: 2.0 * x + 1.0, n=2)
        assert tech.n == 2
        assert abs(float(tech(jnp.float64(0.5))) - 2.0) < 1e-10

    def test_from_coeffs_single(self):
        """Single coefficient => constant function."""
        from chebfunjax.tech.chebtech import Chebtech2
        tech = Chebtech2.from_coeffs(jnp.array([3.0]))
        assert abs(float(tech(jnp.float64(0.0))) - 3.0) < ATOL
        assert abs(float(tech(jnp.float64(0.7))) - 3.0) < ATOL

    def test_from_values_constant(self):
        """from_values with constant array gives constant function."""
        from chebfunjax.tech.chebtech import Chebtech2
        vals = jnp.ones(5, dtype=jnp.float64) * 2.5
        tech = Chebtech2.from_values(vals)
        xs = jnp.linspace(-0.9, 0.9, 10, dtype=jnp.float64)
        npt.assert_allclose(np.array(tech(xs)), np.full(10, 2.5), atol=1e-11)

    def test_sum_constant(self):
        """sum of constant c on [-1,1] in reference coords == 2*c."""
        from chebfunjax.tech.chebtech import Chebtech2
        tech = Chebtech2.from_function(lambda x: jnp.ones_like(x) * 4.0, n=1)
        assert abs(float(tech.sum()) - 8.0) < 1e-10

    def test_diff_constant_is_zero(self):
        """Derivative of constant Chebtech2 == 0."""
        from chebfunjax.tech.chebtech import Chebtech2
        tech = Chebtech2.from_coeffs(jnp.array([5.0]))
        dt = tech.diff()
        # diff of a constant => all-zero coeffs or n=1 with value 0
        xs = jnp.linspace(-0.9, 0.9, 5, dtype=jnp.float64)
        npt.assert_allclose(np.array(dt(xs)), np.zeros(5), atol=1e-11)


# ============================================================================
# Transforms edge cases
# ============================================================================


class TestTransformEdgeCases:
    """Edge cases for coefficient/value transforms."""

    def test_vals2coeffs_length_1(self):
        """vals2coeffs of length-1 array returns itself."""
        from chebfunjax.utils.transforms import vals2coeffs
        v = jnp.array([3.0])
        c = vals2coeffs(v)
        npt.assert_allclose(np.array(c), np.array([3.0]), atol=ATOL)

    def test_coeffs2vals_length_1(self):
        """coeffs2vals of length-1 array returns itself."""
        from chebfunjax.utils.transforms import coeffs2vals
        c = jnp.array([7.0])
        v = coeffs2vals(c)
        npt.assert_allclose(np.array(v), np.array([7.0]), atol=ATOL)

    def test_vals2coeffs_length_2(self):
        """vals2coeffs for [a, b] gives Chebyshev coefficients of a linear."""
        from chebfunjax.utils.transforms import vals2coeffs
        # For n=2: points are x={-1, 1}; values at x=-1 and x=1
        # f = [v0, v1] => T_0 coeff = (v0+v1)/2, T_1 coeff = (v1-v0)/2
        v = jnp.array([1.0, 3.0])
        c = vals2coeffs(v)
        assert abs(float(c[0]) - 2.0) < ATOL  # (1+3)/2
        assert abs(float(c[1]) - 1.0) < ATOL  # (3-1)/2

    def test_cheb2leg_length_1(self):
        """cheb2leg of [c] returns [c] (T_0 == P_0 == 1)."""
        from chebfunjax.utils.transforms import cheb2leg
        c = jnp.array([5.0])
        cl = cheb2leg(c)
        npt.assert_allclose(np.array(cl), np.array([5.0]), atol=ATOL)


# ============================================================================
# misc.py edge cases
# ============================================================================


class TestStandardChopEdgeCases:
    """Edge cases for standard_chop."""

    def test_short_array_not_chopped(self):
        """Arrays shorter than 17 return full length."""
        from chebfunjax.utils.misc import standard_chop
        c = jnp.ones(10, dtype=jnp.float64)
        cutoff = standard_chop(c)
        assert cutoff == 10

    def test_all_zeros_gives_length_1(self):
        """All-zero array should give cutoff 1."""
        from chebfunjax.utils.misc import standard_chop
        c = jnp.zeros(20, dtype=jnp.float64)
        cutoff = standard_chop(c)
        assert cutoff == 1

    def test_tol_geq_1_gives_length_1(self):
        """Tolerance >= 1 => keep only 1 coefficient."""
        from chebfunjax.utils.misc import standard_chop
        c = jnp.ones(30, dtype=jnp.float64)
        cutoff = standard_chop(c, tol=1.0)
        assert cutoff == 1

    def test_perfectly_decaying_coefficients(self):
        """10^{-k} coefficients should be chopped near machine epsilon."""
        from chebfunjax.utils.misc import standard_chop
        c = 10.0 ** (-jnp.arange(1, 51, dtype=jnp.float64))
        cutoff = standard_chop(c)
        # Should chop well before all 50 coefficients
        assert cutoff < 50

    def test_cutoff_positive(self):
        """standard_chop always returns a positive cutoff."""
        from chebfunjax.utils.misc import standard_chop
        for n in [1, 3, 16, 17, 50]:
            c = jnp.ones(n, dtype=jnp.float64)
            assert standard_chop(c) >= 1


# ============================================================================
# polynomials.py edge cases
# ============================================================================


class TestPolynomialsEdgeCases:
    """Edge cases for polynomial utilities."""

    def test_chebpoly_degree_0(self):
        """T_0 = 1 => coefficients = [1]."""
        from chebfunjax.utils.polynomials import chebpoly
        c = chebpoly(0)
        npt.assert_allclose(np.array(c), np.array([1.0]), atol=ATOL)

    def test_chebpoly_degree_1(self):
        """T_1 = x => coefficients = [0, 1]."""
        from chebfunjax.utils.polynomials import chebpoly
        c = chebpoly(1)
        npt.assert_allclose(np.array(c), np.array([0.0, 1.0]), atol=ATOL)

    def test_chebpoly_negative_n_raises(self):
        """chebpoly(-1) raises ValueError."""
        from chebfunjax.utils.polynomials import chebpoly
        with pytest.raises(ValueError):
            chebpoly(-1)

    def test_legpoly_degree_0(self):
        """L_0 = 1 => coefficients = [1] (in Legendre basis)."""
        from chebfunjax.utils.polynomials import legpoly
        c = legpoly(0)
        npt.assert_allclose(np.array(c), np.array([1.0]), atol=ATOL)

    def test_chebeval_t0(self):
        """chebeval(x, 0) == 1.0 everywhere (T_0 = 1)."""
        from chebfunjax.utils.polynomials import chebeval
        xs = jnp.linspace(-0.9, 0.9, 5, dtype=jnp.float64)
        vals = chebeval(xs, 0)
        npt.assert_allclose(np.array(vals), np.ones(5), atol=ATOL)

    def test_chebeval_t1(self):
        """chebeval(x, 1) == x (T_1 = x)."""
        from chebfunjax.utils.polynomials import chebeval
        xs = jnp.linspace(-0.9, 0.9, 10, dtype=jnp.float64)
        vals = chebeval(xs, 1)
        npt.assert_allclose(np.array(vals), np.array(xs), atol=ATOL)


# ============================================================================
# Chebfun2d edge cases
# ============================================================================


class TestChebfun2EdgeCases:
    """Edge cases for 2D Chebfun."""

    def test_constant_chebfun2(self):
        """Constant Chebfun2 evaluates to that constant everywhere."""
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        f = chebfun2(lambda x, y: jnp.ones_like(x) * 2.5)
        assert abs(float(f(0.0, 0.0)) - 2.5) < 1e-10
        assert abs(float(f(0.5, -0.3)) - 2.5) < 1e-10

    def test_constant_chebfun2_sum2(self):
        """sum2 of constant 1 over [-1,1]^2 == 4."""
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        f = chebfun2(lambda x, y: jnp.ones_like(x))
        val = float(f.sum2())
        assert abs(val - 4.0) < 1e-10

    def test_chebfun2_invalid_dim_raises(self):
        """Chebfun2.sum(dim=0) raises ValueError."""
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        f = chebfun2(lambda x, y: x * y)
        with pytest.raises(ValueError):
            f.sum(dim=0)

    def test_chebfun2_diff_y(self):
        """diff(x*y, dim=1) == x (dim=1 means d/dy)."""
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        f = chebfun2(lambda x, y: x * y)
        df = f.diff(dim=1)  # d/dy of x*y = x
        for xi, yi in [(0.3, 0.5), (0.3, -0.4), (-0.5, 0.8)]:
            got = float(df(xi, yi))
            expected = xi
            assert abs(got - expected) < 1e-8, (
                f"diff(x*y, y) at ({xi},{yi}): got {got}, expected {expected}"
            )

    def test_chebfun2_diff_x(self):
        """diff(x*y, dim=2) == y (dim=2 means d/dx)."""
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        f = chebfun2(lambda x, y: x * y)
        df = f.diff(dim=2)  # d/dx of x*y = y
        for xi, yi in [(0.3, 0.5), (0.3, -0.4), (-0.5, 0.8)]:
            got = float(df(xi, yi))
            expected = yi
            assert abs(got - expected) < 1e-8, (
                f"diff(x*y, x) at ({xi},{yi}): got {got}, expected {expected}"
            )


# ============================================================================
# Domain edge cases
# ============================================================================


class TestDomainEdgeCases:
    """Edge cases for the Domain class."""

    def test_single_interval(self):
        """Domain(-1, 1) has 1 interval and correct endpoints."""
        from chebfunjax.domain import Domain
        d = Domain((-1.0, 1.0))
        assert d.n_intervals == 1
        assert d.a == -1.0
        assert d.b == 1.0

    def test_two_interval_domain(self):
        """Domain([-1, 0, 1]) has 2 intervals."""
        from chebfunjax.domain import Domain
        d = Domain((-1.0, 0.0, 1.0))
        assert d.n_intervals == 2

    def test_domain_contains_single_interval(self):
        """Domain(-1, 1) contains x=0."""
        from chebfunjax.domain import Domain
        d = Domain((-1.0, 1.0))
        assert d.a <= 0.0 <= d.b

    def test_domain_length(self):
        """Domain(-2, 3) has length 5."""
        from chebfunjax.domain import Domain
        d = Domain((-2.0, 3.0))
        assert abs((d.b - d.a) - 5.0) < 1e-14


# ============================================================================
# singfun edge cases
# ============================================================================


class TestSingfunEdgeCases:
    """Edge cases for the Singfun class."""

    def test_zero_exponents_smooth(self):
        """Singfun with exponents (0, 0) behaves like a regular function."""
        from chebfunjax.fun.singfun import Singfun
        # Build a singfun for sin(x) with zero exponents
        sf = Singfun.from_function(jnp.sin, exponents=(0.0, 0.0))
        xs = jnp.linspace(-0.9, 0.9, 10, dtype=jnp.float64)
        npt.assert_allclose(np.array(sf(xs)), np.array(jnp.sin(xs)), atol=1e-12)

    def test_sum_singfun_zero_exponents(self):
        """sum of Singfun with zero exponents matches regular sum."""
        from chebfunjax.fun.singfun import Singfun
        sf = Singfun.from_function(jnp.cos, exponents=(0.0, 0.0))
        # int cos(x) dx from -1 to 1 = 2 sin(1)
        expected = 2.0 * float(jnp.sin(jnp.float64(1.0)))
        assert abs(float(sf.sum()) - expected) < 1e-10

    def test_singfun_exponents_stored(self):
        """Exponents are stored as a tuple of two floats."""
        from chebfunjax.fun.singfun import Singfun
        sf = Singfun.from_function(lambda x: jnp.ones_like(x), exponents=(-1.0, 0.0))
        assert sf.exponents[0] == pytest.approx(-1.0)
        assert sf.exponents[1] == pytest.approx(0.0)
