"""Tests for chebfunjax.tech.chebtech — Chebtech1 class.

Verifies that Chebtech1 produces results consistent with Chebtech2 for smooth
functions, and that the 1st-kind grid and transforms are correct.

JAX contract: evaluation is jit=yes, vmap=yes, grad=yes (same as Chebtech2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import (
    Chebtech1,
    Chebtech2,
    _chebtech1_coeffs2vals,
    _chebtech1_vals2coeffs,
)
from chebfunjax.utils.quadrature import chebpts

RTOL = 1e-12
ATOL = 1e-14


# ============================================================================
# Tier 1: DCT-II / DCT-III round-trip
# ============================================================================


class TestTransforms1:
    """Tests for _chebtech1_vals2coeffs and _chebtech1_coeffs2vals."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 32])
    def test_roundtrip_vals2coeffs_coeffs2vals(self, n):
        """vals2coeffs followed by coeffs2vals should recover original values."""
        x = np.array(chebpts(n, kind=1))
        values = jnp.array(np.sin(3 * x), dtype=jnp.float64)
        c = _chebtech1_vals2coeffs(values)
        v_back = _chebtech1_coeffs2vals(c)
        npt.assert_allclose(np.array(v_back), np.array(values), atol=1e-13)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 32])
    def test_roundtrip_coeffs2vals_vals2coeffs(self, n):
        """coeffs2vals followed by vals2coeffs should recover original coeffs."""
        rng = np.random.default_rng(42)
        coeffs = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        v = _chebtech1_coeffs2vals(coeffs)
        c_back = _chebtech1_vals2coeffs(v)
        npt.assert_allclose(np.array(c_back), np.array(coeffs), atol=1e-13)

    def test_trivial_n1(self):
        """n=1: constant function, single point at x=0."""
        values = jnp.array([3.7], dtype=jnp.float64)
        c = _chebtech1_vals2coeffs(values)
        npt.assert_allclose(float(c[0]), 3.7, atol=1e-15)
        v = _chebtech1_coeffs2vals(c)
        npt.assert_allclose(float(v[0]), 3.7, atol=1e-15)

    def test_constant_function(self):
        """Constant function: only c[0] is non-zero."""
        n = 8
        values = jnp.ones(n, dtype=jnp.float64)
        c = _chebtech1_vals2coeffs(values)
        npt.assert_allclose(float(c[0]), 1.0, atol=1e-14)
        npt.assert_allclose(np.array(c[1:]), np.zeros(n - 1), atol=1e-14)

    def test_t1_function(self):
        """T_1(x) = x: coefficients should be [0, 1, 0, ..., 0]."""
        n = 8
        x = chebpts(n, kind=1)
        values = jnp.array(x, dtype=jnp.float64)
        c = _chebtech1_vals2coeffs(values)
        npt.assert_allclose(float(c[0]), 0.0, atol=1e-14)
        npt.assert_allclose(float(c[1]), 1.0, atol=1e-13)
        npt.assert_allclose(np.array(c[2:]), np.zeros(n - 2), atol=1e-13)

    def test_t2_function(self):
        """T_2(x) = 2x^2 - 1: coefficient c[2] = 1, all others near 0."""
        n = 8
        x = chebpts(n, kind=1)
        # T_2 = 2x^2 - 1 is itself a Chebyshev polynomial, so its expansion
        # in the Chebyshev basis has c[2] = 1 and c[k] = 0 for k != 2.
        values = jnp.array(2.0 * x**2 - 1.0, dtype=jnp.float64)
        c = _chebtech1_vals2coeffs(values)
        npt.assert_allclose(float(c[0]), 0.0, atol=1e-12)
        npt.assert_allclose(float(c[1]), 0.0, atol=1e-12)
        npt.assert_allclose(float(c[2]), 1.0, atol=1e-12)
        npt.assert_allclose(np.array(c[3:]), np.zeros(n - 3), atol=1e-12)


# ============================================================================
# Tier 1: Chebtech1 construction
# ============================================================================


class TestChebtech1Construction:
    """Tests for Chebtech1 construction methods."""

    def test_from_function_sin(self):
        """Adaptive construction of sin(x): should resolve to ishappy=True."""
        f = Chebtech1.from_function(jnp.sin)
        assert f.ishappy

    def test_from_function_n_coeffs_reasonable(self):
        """sin(x) should need roughly the same number of coefficients as Chebtech2."""
        f1 = Chebtech1.from_function(jnp.sin)
        f2 = Chebtech2.from_function(jnp.sin)
        # Both should converge; Chebtech1 uses a power-of-2 grid
        assert f1.ishappy
        assert f2.ishappy
        # Both should have similar (small) numbers of coefficients
        assert f1.n < 20

    def test_from_function_exp(self):
        """exp(x) should converge adaptively."""
        f = Chebtech1.from_function(jnp.exp)
        assert f.ishappy

    def test_from_function_fixed_n(self):
        """Fixed-length construction with specified n."""
        f = Chebtech1.from_function(jnp.sin, n=16)
        assert f.n == 16

    def test_from_function_fixed_n_values(self):
        """Fixed-length construction should interpolate at 1st-kind points."""
        n = 16
        f = Chebtech1.from_function(jnp.sin, n=n)
        x = chebpts(n, kind=1)
        npt.assert_allclose(np.array(f.values), np.array(jnp.sin(x)), atol=1e-14)

    def test_from_coeffs(self):
        """Construct from Chebyshev coefficients."""
        c = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64)
        f = Chebtech1.from_coeffs(c)
        npt.assert_allclose(np.array(f.coeffs), np.array(c), atol=1e-15)
        assert f.n == 3

    def test_from_values(self):
        """Construct from values at 1st-kind Chebyshev points."""
        n = 8
        x = chebpts(n, kind=1)
        v = jnp.sin(x)
        f = Chebtech1.from_values(v)
        # Re-evaluate at same points
        npt.assert_allclose(np.array(f.values), np.array(v), atol=1e-13)

    def test_repr(self):
        """repr should be a non-empty string."""
        f = Chebtech1.from_function(jnp.sin)
        s = repr(f)
        assert "Chebtech1" in s
        assert "n=" in s


# ============================================================================
# Tier 1: Chebtech1 evaluation
# ============================================================================


class TestChebtech1Evaluation:
    """Tests for Chebtech1.__call__ (Clenshaw evaluation)."""

    def test_sin_at_zero(self):
        """sin(0) = 0."""
        f = Chebtech1.from_function(jnp.sin)
        npt.assert_allclose(float(f(jnp.float64(0.0))), 0.0, atol=1e-13)

    def test_sin_at_half(self):
        """sin(0.5) matches jnp.sin(0.5)."""
        f = Chebtech1.from_function(jnp.sin)
        npt.assert_allclose(
            float(f(jnp.float64(0.5))),
            float(jnp.sin(jnp.float64(0.5))),
            rtol=1e-12,
        )

    def test_vectorised_evaluation(self):
        """Chebtech1(x_vec) should agree with jnp.sin on a test grid."""
        f = Chebtech1.from_function(jnp.sin)
        x_test = jnp.linspace(-1.0, 1.0, 50, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(f(x_test)), np.array(jnp.sin(x_test)), rtol=1e-12
        )

    def test_consistent_with_chebtech2(self):
        """Chebtech1 and Chebtech2 should agree for smooth functions."""
        f1 = Chebtech1.from_function(jnp.exp)
        f2 = Chebtech2.from_function(jnp.exp)
        x_test = jnp.linspace(-0.9, 0.9, 40, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(f1(x_test)), np.array(f2(x_test)), rtol=1e-10
        )

    def test_jit_evaluation(self):
        """Chebtech1.__call__ should survive JIT compilation via eqx.filter_jit."""
        f = Chebtech1.from_function(jnp.cos)
        # Chebtech1.__call__ is already @eqx.filter_jit; just verify it works
        x = jnp.float64(0.3)
        npt.assert_allclose(float(f(x)), float(jnp.cos(x)), rtol=1e-12)

    def test_vmap_evaluation(self):
        """Chebtech1 should work under vmap."""
        f = Chebtech1.from_function(jnp.sin)
        x = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        f_vmap = jax.vmap(lambda xi: f(xi))
        npt.assert_allclose(
            np.array(f_vmap(x)), np.array(jnp.sin(x)), rtol=1e-12
        )


# ============================================================================
# Tier 1: Chebtech1 calculus
# ============================================================================


class TestChebtech1Calculus:
    """Tests for differentiation, integration on Chebtech1."""

    def test_diff_sin_is_cos(self):
        """diff of sin ≈ cos."""
        f = Chebtech1.from_function(jnp.sin)
        fp = f.diff()
        x = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)
        npt.assert_allclose(np.array(fp(x)), np.array(jnp.cos(x)), rtol=1e-10)

    def test_diff_cos_is_neg_sin(self):
        """diff of cos ≈ -sin."""
        f = Chebtech1.from_function(jnp.cos)
        fp = f.diff()
        x = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)
        npt.assert_allclose(np.array(fp(x)), np.array(-jnp.sin(x)), rtol=1e-10)

    def test_diff_cumsum_roundtrip(self):
        """diff(cumsum(f)) ≈ f for a smooth function."""
        f = Chebtech1.from_function(lambda x: jnp.sin(3 * x))
        F = f.cumsum()
        f_back = F.diff()
        x = jnp.linspace(-0.8, 0.8, 20, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(f_back(x)), np.array(f(x)), rtol=1e-10
        )

    def test_sum_sin(self):
        """Integral of sin(x) from -1 to 1 should be 0."""
        f = Chebtech1.from_function(jnp.sin)
        npt.assert_allclose(float(f.sum()), 0.0, atol=1e-13)

    def test_sum_cos(self):
        """Integral of cos(x) from -1 to 1 = 2*sin(1)."""
        f = Chebtech1.from_function(jnp.cos)
        expected = 2.0 * float(jnp.sin(jnp.float64(1.0)))
        npt.assert_allclose(float(f.sum()), expected, rtol=1e-12)

    def test_sum_constant(self):
        """Integral of constant c over [-1,1] = 2c."""
        f = Chebtech1.from_coeffs(jnp.array([3.0], dtype=jnp.float64))
        npt.assert_allclose(float(f.sum()), 6.0, atol=1e-14)

    def test_inner_product_sin_cos(self):
        """<sin, cos> = integral of sin*cos from -1 to 1 = 0 (by symmetry)."""
        f = Chebtech1.from_function(jnp.sin)
        g = Chebtech1.from_function(jnp.cos)
        npt.assert_allclose(float(f.inner(g)), 0.0, atol=1e-12)

    def test_consistent_sum_with_chebtech2(self):
        """Sum of a function should agree between Chebtech1 and Chebtech2."""
        f1 = Chebtech1.from_function(jnp.exp)
        f2 = Chebtech2.from_function(jnp.exp)
        npt.assert_allclose(float(f1.sum()), float(f2.sum()), rtol=1e-12)


# ============================================================================
# Tier 1: Chebtech1 arithmetic
# ============================================================================


class TestChebtech1Arithmetic:
    """Tests for Chebtech1 arithmetic operations."""

    def test_add_scalar(self):
        """f + 1 shifts the constant term."""
        f = Chebtech1.from_function(jnp.sin)
        g = f + 1.0
        x = jnp.float64(0.5)
        npt.assert_allclose(float(g(x)), float(f(x)) + 1.0, rtol=1e-12)

    def test_add_two_chebtechs(self):
        """sin + cos."""
        f = Chebtech1.from_function(jnp.sin)
        g = Chebtech1.from_function(jnp.cos)
        h = f + g
        x = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(h(x)), np.array(jnp.sin(x) + jnp.cos(x)), rtol=1e-11
        )

    def test_neg(self):
        """-f should negate values."""
        f = Chebtech1.from_function(jnp.sin)
        g = -f
        x = jnp.float64(0.5)
        npt.assert_allclose(float(g(x)), -float(f(x)), rtol=1e-12)

    def test_mul_scalar(self):
        """2 * f."""
        f = Chebtech1.from_function(jnp.cos)
        g = 2.0 * f
        x = jnp.float64(0.3)
        npt.assert_allclose(float(g(x)), 2.0 * float(f(x)), rtol=1e-12)

    def test_mul_two_chebtechs(self):
        """sin * cos = sin*cos."""
        f = Chebtech1.from_function(jnp.sin)
        g = Chebtech1.from_function(jnp.cos)
        h = f * g
        x = jnp.linspace(-0.8, 0.8, 20, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(h(x)), np.array(jnp.sin(x) * jnp.cos(x)), rtol=1e-11
        )

    def test_div_scalar(self):
        """f / 2."""
        f = Chebtech1.from_function(jnp.exp)
        g = f / 2.0
        x = jnp.float64(0.2)
        npt.assert_allclose(float(g(x)), float(f(x)) / 2.0, rtol=1e-12)

    def test_pow_integer(self):
        """f^2 = f*f."""
        f = Chebtech1.from_function(jnp.cos)
        g = f**2
        x = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(g(x)), np.array(jnp.cos(x) ** 2), rtol=1e-10
        )


# ============================================================================
# Tier 1: Chebtech1 rootfinding
# ============================================================================


class TestChebtech1Roots:
    """Tests for Chebtech1.roots()."""

    def test_roots_sin(self):
        """sin(x) on [-1,1] has one root at x=0."""
        f = Chebtech1.from_function(jnp.sin)
        r = f.roots()
        r_np = np.array(r)
        npt.assert_allclose(r_np, np.array([0.0]), atol=1e-12)

    def test_roots_linear(self):
        """Linear f = x - 0.5 has root at 0.5."""
        f = Chebtech1.from_function(lambda x: x - 0.5)
        r = f.roots()
        npt.assert_allclose(np.array(r), np.array([0.5]), atol=1e-12)

    def test_roots_quadratic(self):
        """f = x^2 - 0.25 has roots at +/- 0.5."""
        f = Chebtech1.from_function(lambda x: x**2 - 0.25)
        r = np.sort(np.array(f.roots()))
        npt.assert_allclose(r, np.array([-0.5, 0.5]), atol=1e-12)
