"""Mathematical invariant tests (V24-V25).

Tests the following invariants:

V24 — Calculus identities:
  - diff(cumsum(f)) ≈ f  (Fundamental Theorem of Calculus)
  - cumsum(diff(f)) ≈ f - f(a)  (antiderivative)
  - sum(f * g) ≈ inner(f, g)  (inner product consistency)
  - sum(f) == integral of f
  - norm(f, 2)^2 ≈ inner(f, f)

V25 — Coefficient transform round-trips:
  - cheb2leg(leg2cheb(c)) ≈ c
  - leg2cheb(cheb2leg(c)) ≈ c
  - cheb2jac(jac2cheb(c)) ≈ c
  - vals2coeffs(coeffs2vals(c)) ≈ c
  - coeffs2vals(vals2coeffs(v)) ≈ v
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import (
    cheb2jac,
    cheb2leg,
    coeffs2vals,
    jac2cheb,
    leg2cheb,
    vals2coeffs,
)

# ============================================================================
# Helpers
# ============================================================================

ATOL_TIGHT = 1e-11
ATOL_LOOSE = 1e-9


def _eval_pts(n=40, a=-1.0, b=1.0):
    """Interior evaluation points (away from endpoints)."""
    return jnp.linspace(a + 1e-4, b - 1e-4, n, dtype=jnp.float64)


# ============================================================================
# V24 — Calculus identities
# ============================================================================


class TestDiffCumsum:
    """Test that diff(cumsum(f)) ≈ f and cumsum(diff(f)) ≈ f - f(a)."""

    def test_diff_cumsum_sin(self):
        """diff(cumsum(sin)) == sin."""
        f = chebfun(jnp.sin)
        g = f.cumsum().diff()
        xs = _eval_pts()
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), atol=ATOL_TIGHT)

    def test_diff_cumsum_exp(self):
        """diff(cumsum(exp)) == exp."""
        f = chebfun(jnp.exp)
        g = f.cumsum().diff()
        xs = _eval_pts()
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), atol=ATOL_TIGHT)

    def test_diff_cumsum_polynomial(self):
        """diff(cumsum(x^3)) == x^3."""
        f = chebfun(lambda x: x ** 3)
        g = f.cumsum().diff()
        xs = _eval_pts()
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), atol=ATOL_TIGHT)

    def test_cumsum_diff_sin(self):
        """cumsum(diff(sin)) == sin - sin(-1)."""
        f = chebfun(jnp.sin)
        a = float(f.domain.a)
        g = f.diff().cumsum()
        # cumsum(f') = f - f(a)
        xs = _eval_pts()
        expected = np.array(f(xs)) - float(f(jnp.float64(a)))
        npt.assert_allclose(np.array(g(xs)), expected, atol=ATOL_TIGHT)

    def test_cumsum_diff_exp(self):
        """cumsum(diff(exp)) == exp - exp(-1)."""
        f = chebfun(jnp.exp)
        a = float(f.domain.a)
        g = f.diff().cumsum()
        xs = _eval_pts()
        expected = np.array(f(xs)) - float(f(jnp.float64(a)))
        npt.assert_allclose(np.array(g(xs)), expected, atol=ATOL_TIGHT)

    def test_diff_cumsum_on_subinterval(self):
        """diff(cumsum(f)) == f on non-default domain."""
        f = chebfun(jnp.cos, domain=(0.0, 2.0))
        g = f.cumsum().diff()
        xs = _eval_pts(a=0.0, b=2.0)
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), atol=ATOL_TIGHT)

    def test_diff_cumsum_high_order(self):
        """diff(diff(cumsum(cumsum(f)))) == f (second-order FTC)."""
        f = chebfun(lambda x: jnp.sin(3.0 * x) * jnp.cos(x))
        g = f.cumsum().cumsum().diff().diff()
        xs = _eval_pts()
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), atol=ATOL_LOOSE)

    def test_diff_order_2_cumsum_twice(self):
        """diff(2)(cumsum(cumsum(f))) == f."""
        f = chebfun(lambda x: jnp.exp(-x ** 2) * jnp.cos(5.0 * x))
        g = f.cumsum().cumsum().diff(2)
        xs = _eval_pts()
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), atol=ATOL_LOOSE)


class TestSumInnerProduct:
    """Test consistency between sum(f*g) and inner(f, g)."""

    def test_sum_fg_equals_inner_fg_sin_cos(self):
        """sum(sin * cos) == inner(sin, cos)."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        s = float(f.sum())  # noqa — here we use sum on f*g
        product_sum = float((f * g).sum())
        inner_val = float(f.inner(g))
        assert abs(product_sum - inner_val) < ATOL_TIGHT

    def test_sum_fg_equals_inner_fg_polynomials(self):
        """sum(x^2 * x^3) == inner(x^2, x^3)."""
        f = chebfun(lambda x: x ** 2)
        g = chebfun(lambda x: x ** 3)
        product_sum = float((f * g).sum())
        inner_val = float(f.inner(g))
        assert abs(product_sum - inner_val) < ATOL_TIGHT

    def test_inner_self_positive(self):
        """inner(f, f) > 0 for any nonzero f."""
        f = chebfun(lambda x: jnp.sin(jnp.pi * x))
        assert float(f.inner(f)) > 0.0

    def test_norm2_sq_equals_inner_self(self):
        """norm(f, 2)^2 == inner(f, f)."""
        f = chebfun(jnp.exp)
        inner = float(f.inner(f))
        norm2_sq = float(f.norm(2)) ** 2
        assert abs(norm2_sq - inner) < ATOL_TIGHT

    def test_inner_linearity_in_second_arg(self):
        """inner(f, g + h) == inner(f, g) + inner(f, h)."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        h = chebfun(jnp.exp)
        lhs = float(f.inner(g + h))
        rhs = float(f.inner(g)) + float(f.inner(h))
        assert abs(lhs - rhs) < ATOL_TIGHT

    def test_sum_equals_integral(self):
        """sum(sin) == integral of sin on [-1, 1] = 0."""
        f = chebfun(jnp.sin)
        assert abs(float(f.sum())) < ATOL_TIGHT  # sin is odd on [-1, 1]

    def test_sum_cos_is_correct(self):
        """sum(cos) == 2 * sin(1)."""
        f = chebfun(jnp.cos)
        expected = 2.0 * float(jnp.sin(jnp.float64(1.0)))
        assert abs(float(f.sum()) - expected) < ATOL_TIGHT


# ============================================================================
# V25 — Coefficient transform round-trips
# ============================================================================


class TestCheb2Leg2Cheb:
    """Round-trip: cheb2leg(leg2cheb(c)) ≈ c."""

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_cheb2leg_leg2cheb_roundtrip(self, n):
        """cheb2leg(leg2cheb(c)) ≈ c for random coefficient vector."""
        rng = np.random.default_rng(42)
        c = jnp.array(rng.standard_normal(n))
        c_roundtrip = cheb2leg(leg2cheb(c))
        npt.assert_allclose(np.array(c_roundtrip), np.array(c), atol=ATOL_LOOSE)

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_leg2cheb_cheb2leg_roundtrip(self, n):
        """leg2cheb(cheb2leg(c)) ≈ c for random coefficient vector."""
        rng = np.random.default_rng(123)
        c = jnp.array(rng.standard_normal(n))
        c_roundtrip = leg2cheb(cheb2leg(c))
        npt.assert_allclose(np.array(c_roundtrip), np.array(c), atol=ATOL_LOOSE)

    def test_cheb2leg_t0(self):
        """T_0 = 1 should map to L_0 = 1 in Legendre basis."""
        c = jnp.array([1.0])
        cl = cheb2leg(c)
        npt.assert_allclose(np.array(cl), np.array([1.0]), atol=ATOL_TIGHT)

    def test_cheb2leg_t1(self):
        """T_1 = x should map to L_1 = x in Legendre basis."""
        c = jnp.array([0.0, 1.0])
        cl = cheb2leg(c)
        # T_1 = x = L_1 in both bases
        npt.assert_allclose(np.array(cl), np.array([0.0, 1.0]), atol=ATOL_TIGHT)

    def test_cheb2leg_consistent_with_evaluation(self):
        """leg2cheb(cheb2leg(c)) == c round-trip is consistent."""
        rng = np.random.default_rng(7)
        n = 10
        c_cheb = jnp.array(rng.standard_normal(n))
        c_leg = cheb2leg(c_cheb)
        # Convert Legendre coefficients back to Cheb: should recover original
        c_cheb_from_leg = leg2cheb(c_leg)
        npt.assert_allclose(np.array(c_cheb_from_leg), np.array(c_cheb), atol=ATOL_LOOSE)


class TestCheb2Jac2Cheb:
    """Round-trip for Chebyshev <-> Jacobi transforms."""

    @pytest.mark.parametrize("alpha,beta", [(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5)])
    def test_cheb2jac_jac2cheb_roundtrip(self, alpha, beta):
        """cheb2jac(jac2cheb(c, alpha, beta), alpha, beta) ≈ c."""
        rng = np.random.default_rng(99)
        n = 12
        c = jnp.array(rng.standard_normal(n))
        c_j = cheb2jac(c, alpha, beta)
        c_back = jac2cheb(c_j, alpha, beta)
        npt.assert_allclose(np.array(c_back), np.array(c), atol=ATOL_LOOSE)


class TestVals2Coeffs2Vals:
    """Round-trip for Chebyshev values <-> coefficients transforms."""

    @pytest.mark.parametrize("n", [3, 8, 17, 32])
    def test_coeffs2vals_vals2coeffs_roundtrip(self, n):
        """coeffs2vals(vals2coeffs(v)) ≈ v."""
        rng = np.random.default_rng(55)
        v = jnp.array(rng.standard_normal(n))
        v_roundtrip = coeffs2vals(vals2coeffs(v))
        npt.assert_allclose(np.array(v_roundtrip), np.array(v), atol=ATOL_TIGHT)

    @pytest.mark.parametrize("n", [3, 8, 17, 32])
    def test_vals2coeffs_coeffs2vals_roundtrip(self, n):
        """vals2coeffs(coeffs2vals(c)) ≈ c."""
        rng = np.random.default_rng(66)
        c = jnp.array(rng.standard_normal(n))
        c_roundtrip = vals2coeffs(coeffs2vals(c))
        npt.assert_allclose(np.array(c_roundtrip), np.array(c), atol=ATOL_TIGHT)

    def test_vals2coeffs_constant(self):
        """vals2coeffs of all-ones should give [1, 0, 0, ...]."""
        n = 5
        v = jnp.ones(n, dtype=jnp.float64)
        c = vals2coeffs(v)
        # T_0 = 1 everywhere => coefficient vector = [1, 0, 0, ...]
        npt.assert_allclose(float(c[0]), 1.0, atol=ATOL_TIGHT)
        npt.assert_allclose(np.array(c[1:]), np.zeros(n - 1), atol=ATOL_TIGHT)

    def test_vals2coeffs_chebyshev_polynomial(self):
        """vals2coeffs of T_3 values should give [0, 0, 0, 1]."""
        n = 4
        x = chebpts(n, kind=2)
        v = jnp.cos(3.0 * jnp.arccos(x))  # T_3(x)
        c = vals2coeffs(v)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        npt.assert_allclose(np.array(c), expected, atol=ATOL_TIGHT)
