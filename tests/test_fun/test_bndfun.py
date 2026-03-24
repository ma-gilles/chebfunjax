"""Tests for chebfunjax.fun.bndfun — Bndfun and Classicfun.

Tests the bounded-interval function representation (layer 3 of the
architecture: Fun on [a, b] wrapping a Chebtech2 on [-1, 1]).

JAX contract:
- __call__: jit=YES, vmap=YES, grad=YES
- diff, cumsum, sum, inner, norm: jit=YES (for the result); construction NOT jit
- roots, min, max, minandmax: jit=NO (variable output / eigenvalue problem)
- Arithmetic operators: jit=YES for the result; construction NOT jit

Test domains:
- [0, π]     — sin(x), ∫ sin = 2, d/dx sin = cos
- [1, 3]     — exp(x), ∫ exp dx = e³ - e, d/dx exp = exp
- [-2, 5]    — x², ∫ x² dx = 5³/3 - (-2)³/3 = 41
- [-1, 1]    — standard domain (regression: should match Chebtech2 directly)
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.classicfun import Classicfun
from chebfunjax.tech.chebtech import Chebtech2

# Tolerances from project.conf
RTOL = 1e-12
ATOL = 1e-14

# Common domains
D_01PI = Domain((0.0, float(jnp.pi)))
D_13 = Domain((1.0, 3.0))
D_STD = Domain((-1.0, 1.0))
D_NEG = Domain((-2.0, 5.0))


# ============================================================================
# Tier 1: Construction
# ============================================================================

class TestBndfunConstruction:
    """Tests for Bndfun construction."""

    def test_from_function_standard_domain(self):
        """Bndfun on [-1,1] is equivalent to Chebtech2.from_function."""
        f = Bndfun.from_function(jnp.sin, D_STD)
        t = Chebtech2.from_function(jnp.sin)
        assert f.ishappy
        assert f.n == t.n

    def test_from_function_pi_domain(self):
        """Bndfun on [0, π] resolves sin(x) with ~14 coefficients."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        assert f.ishappy
        # sin on [0, π] — same smoothness as on [-1,1]; expect ~14 coeffs
        assert 10 <= f.n <= 20

    def test_from_function_exp_on_1_3(self):
        """Adaptive construction of exp on [1, 3]."""
        f = Bndfun.from_function(jnp.exp, D_13)
        assert f.ishappy
        assert f.n > 1

    def test_from_function_fixed_n(self):
        """Fixed-n construction."""
        f = Bndfun.from_function(jnp.sin, D_01PI, n=20)
        assert f.n == 20

    def test_from_chebtech(self):
        """from_chebtech wraps an existing Chebtech2."""
        t = Chebtech2.from_function(jnp.sin)
        f = Bndfun.from_chebtech(t, D_STD)
        assert f.onefun is t
        assert f.domain == D_STD

    def test_invalid_multi_interval_domain(self):
        """Bndfun raises ValueError for piecewise domains."""
        d = Domain((-1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="single-interval"):
            Bndfun.from_function(jnp.sin, d)

    def test_isinstance_classicfun(self):
        """Bndfun is a subclass of Classicfun."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        assert isinstance(f, Classicfun)

    def test_domain_attribute(self):
        """domain attribute matches construction domain."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        assert f.domain == D_01PI


# ============================================================================
# Tier 1: Evaluation
# ============================================================================

class TestBndfunEvaluation:
    """Tests for Bndfun evaluation.

    JAX contract: jit=YES, vmap=YES, grad=YES.
    """

    def test_sin_at_midpoint(self):
        """sin(π/2) on [0, π] should be 1."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(f(x)), 1.0, rtol=RTOL, atol=ATOL)

    def test_sin_at_endpoints(self):
        """sin(0) = 0, sin(π) ≈ 0."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        npt.assert_allclose(float(f(jnp.float64(0.0))), 0.0, atol=1e-14)
        npt.assert_allclose(float(f(jnp.float64(float(jnp.pi)))), 0.0, atol=1e-14)

    def test_sin_array(self):
        """Evaluate at many points matches numpy sin."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        xs = jnp.linspace(0, float(jnp.pi), 50, dtype=jnp.float64)
        ys = f(xs)
        # atol=1e-14 handles near-zero endpoint values (sin(0) ~ 1.7e-16 round-off)
        npt.assert_allclose(np.array(ys), np.array(jnp.sin(xs)), atol=1e-14)

    def test_exp_on_1_3(self):
        """Evaluate exp on [1, 3] at several points."""
        f = Bndfun.from_function(jnp.exp, D_13)
        xs = jnp.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(f(xs)), np.array(jnp.exp(xs)), rtol=RTOL)

    def test_standard_domain_matches_chebtech2(self):
        """Bndfun on [-1, 1] evaluates identically to Chebtech2."""
        t = Chebtech2.from_function(jnp.sin)
        f = Bndfun.from_chebtech(t, D_STD)
        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(f(xs)), np.array(t(xs)), rtol=1e-14)

    def test_jit_evaluation(self):
        """Evaluation is JIT-safe."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        fast = jax.jit(lambda x: f(x))
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(fast(x)), 1.0, rtol=RTOL)

    def test_vmap_evaluation(self):
        """Evaluation is vmap-safe."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        xs = jnp.linspace(0, float(jnp.pi), 20, dtype=jnp.float64)
        ys = jax.vmap(f)(xs)
        # atol=1e-14 for near-zero endpoint values
        npt.assert_allclose(np.array(ys), np.array(jnp.sin(xs)), atol=1e-14)

    def test_grad_evaluation(self):
        """Gradient of evaluation w.r.t. input x is JIT/grad-safe."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        # d/dx sin(x) = cos(x)
        df = jax.grad(lambda x: f(x))
        for x0 in [0.5, 1.0, 2.0, float(jnp.pi) - 0.1]:
            x0_arr = jnp.float64(x0)
            npt.assert_allclose(float(df(x0_arr)), float(jnp.cos(x0_arr)),
                                rtol=1e-10)

    def test_filter_jit_with_module_arg(self):
        """eqx.filter_jit passing Bndfun as argument."""
        f = Bndfun.from_function(jnp.sin, D_01PI)

        @eqx.filter_jit
        def evaluate(g, x):
            return g(x)

        x = jnp.float64(1.0)
        npt.assert_allclose(float(evaluate(f, x)), float(jnp.sin(x)), rtol=RTOL)


# ============================================================================
# Tier 1: Arithmetic
# ============================================================================

class TestBndfunArithmetic:
    """Tests for Bndfun arithmetic operators."""

    def test_add_two_bndfuns(self):
        """sin + cos on [0, π]: check at π/4."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = Bndfun.from_function(jnp.cos, D_01PI)
        h = f + g
        x = jnp.float64(jnp.pi / 4)
        expected = float(jnp.sin(x) + jnp.cos(x))
        npt.assert_allclose(float(h(x)), expected, rtol=RTOL)

    def test_add_scalar(self):
        """sin + 1 on [0, π]."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = f + 1.0
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(h(x)), 2.0, rtol=RTOL)

    def test_radd_scalar(self):
        """1 + sin."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = 1.0 + f
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(h(x)), 2.0, rtol=RTOL)

    def test_sub_two_bndfuns(self):
        """sin - cos on [0, π]."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = Bndfun.from_function(jnp.cos, D_01PI)
        h = f - g
        x = jnp.float64(jnp.pi / 4)
        expected = float(jnp.sin(x) - jnp.cos(x))
        # atol=1e-14 since sin(π/4) - cos(π/4) = 0 exactly, round-off expected
        npt.assert_allclose(float(h(x)), expected, atol=1e-14)

    def test_sub_scalar(self):
        """sin - 1."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = f - 1.0
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(h(x)), 0.0, atol=1e-14)

    def test_neg(self):
        """-f."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = -f
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(h(x)), -1.0, rtol=RTOL)

    def test_mul_two_bndfuns(self):
        """sin * sin = sin²."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = f * f
        x = jnp.float64(jnp.pi / 3)
        expected = float(jnp.sin(x) ** 2)
        npt.assert_allclose(float(h(x)), expected, rtol=RTOL)

    def test_mul_scalar(self):
        """2 * sin."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = 2.0 * f
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(h(x)), 2.0, rtol=RTOL)

    def test_div_scalar(self):
        """sin / 2."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = f / 2.0
        x = jnp.float64(jnp.pi / 2)
        npt.assert_allclose(float(h(x)), 0.5, rtol=RTOL)

    def test_pow_integer(self):
        """sin**3 on [0, π]."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        h = f ** 3
        x = jnp.float64(jnp.pi / 6)
        expected = float(jnp.sin(x) ** 3)
        npt.assert_allclose(float(h(x)), expected, rtol=RTOL)

    def test_domain_mismatch_raises(self):
        """Adding Bndfuns with different domains should raise ValueError."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = Bndfun.from_function(jnp.cos, D_13)
        with pytest.raises(ValueError, match="domains do not match"):
            _ = f + g

    def test_arithmetic_returns_bndfun(self):
        """Arithmetic operations preserve the Bndfun type."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = Bndfun.from_function(jnp.cos, D_01PI)
        assert isinstance(f + g, Bndfun)
        assert isinstance(f - g, Bndfun)
        assert isinstance(f * g, Bndfun)
        assert isinstance(f + 1.0, Bndfun)
        assert isinstance(-f, Bndfun)


# ============================================================================
# Tier 1: Calculus
# ============================================================================

class TestBndfunCalculus:
    """Tests for Bndfun diff, cumsum, sum."""

    def test_diff_sin_is_cos(self):
        """d/dx sin(x) = cos(x) on [0, π]."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        fp = f.diff()
        xs = jnp.linspace(0, float(jnp.pi), 30, dtype=jnp.float64)
        npt.assert_allclose(np.array(fp(xs)), np.array(jnp.cos(xs)), rtol=RTOL)

    def test_diff_cos_is_neg_sin(self):
        """d/dx cos(x) = -sin(x) on [0, π]."""
        f = Bndfun.from_function(jnp.cos, D_01PI)
        fp = f.diff()
        xs = jnp.linspace(0.1, float(jnp.pi) - 0.1, 30, dtype=jnp.float64)
        npt.assert_allclose(np.array(fp(xs)), np.array(-jnp.sin(xs)), rtol=RTOL)

    def test_diff_exp_on_1_3(self):
        """d/dx exp(x) = exp(x) on [1, 3]."""
        f = Bndfun.from_function(jnp.exp, D_13)
        fp = f.diff()
        xs = jnp.linspace(1.0, 3.0, 30, dtype=jnp.float64)
        npt.assert_allclose(np.array(fp(xs)), np.array(jnp.exp(xs)), rtol=RTOL)

    def test_diff_polynomial(self):
        """d/dx x² = 2x on [-2, 5]."""
        f = Bndfun.from_function(lambda x: x**2, D_NEG)
        fp = f.diff()
        xs = jnp.linspace(-2.0, 5.0, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(fp(xs)), np.array(2 * xs), rtol=RTOL)

    def test_diff_second_order(self):
        """d²/dx² sin(x) = -sin(x) on [0, π]."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        fpp = f.diff(k=2)
        xs = jnp.linspace(0.1, float(jnp.pi) - 0.1, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(fpp(xs)), np.array(-jnp.sin(xs)), rtol=RTOL)

    def test_diff_zeroth_order(self):
        """diff(k=0) is the identity."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        f0 = f.diff(k=0)
        xs = jnp.linspace(0, float(jnp.pi), 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(f0(xs)), np.array(f(xs)), rtol=1e-14)

    def test_diff_cumsum_is_identity(self):
        """diff(cumsum(f)) ≈ f (fundamental theorem of calculus)."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        F = f.cumsum()
        dF = F.diff()
        xs = jnp.linspace(0.1, float(jnp.pi) - 0.1, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(dF(xs)), np.array(f(xs)), rtol=RTOL)

    def test_cumsum_zero_at_left_endpoint(self):
        """Antiderivative satisfies F(a) = 0."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        F = f.cumsum()
        npt.assert_allclose(float(F(jnp.float64(0.0))), 0.0, atol=1e-13)

    def test_cumsum_sin_is_neg_cos_plus_1(self):
        """∫₀^x sin(t)dt = 1 - cos(x)."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        F = f.cumsum()
        xs = jnp.linspace(0, float(jnp.pi), 20, dtype=jnp.float64)
        expected = 1 - jnp.cos(xs)
        # atol=1e-14 for near-zero values at left endpoint
        npt.assert_allclose(np.array(F(xs)), np.array(expected), atol=1e-14)

    def test_sum_sin_on_0_pi(self):
        """∫₀^π sin(x) dx = 2 (the golden test case)."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        result = float(f.sum())
        npt.assert_allclose(result, 2.0, rtol=RTOL, atol=ATOL)

    def test_sum_exp_on_1_3(self):
        """∫₁^3 exp(x) dx = e³ - e."""
        f = Bndfun.from_function(jnp.exp, D_13)
        expected = float(jnp.exp(3.0) - jnp.exp(1.0))
        npt.assert_allclose(float(f.sum()), expected, rtol=RTOL)

    def test_sum_polynomial_on_neg2_5(self):
        """∫_{-2}^5 x² dx = 5³/3 - (-2)³/3 = 125/3 + 8/3 = 133/3 ≈ 44.333."""
        f = Bndfun.from_function(lambda x: x**2, D_NEG)
        expected = (5.0**3 - (-2.0)**3) / 3.0  # = 44.3333...
        npt.assert_allclose(float(f.sum()), expected, rtol=RTOL)

    def test_sum_constant_one(self):
        """∫_a^b 1 dx = b - a."""
        for domain in [D_01PI, D_13, D_NEG]:
            f = Bndfun.from_function(lambda x: jnp.ones_like(x), domain)
            a, b = domain.a, domain.b
            npt.assert_allclose(float(f.sum()), b - a, rtol=RTOL)

    def test_inner_product_sin_cos(self):
        """⟨sin, cos⟩ over [0, π] = ∫₀^π sin cos dx = 0."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = Bndfun.from_function(jnp.cos, D_01PI)
        result = float(f.inner(g))
        npt.assert_allclose(result, 0.0, atol=1e-12)

    def test_inner_product_sin_sin(self):
        """⟨sin, sin⟩ over [0, π] = ∫₀^π sin² dx = π/2."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        result = float(f.inner(f))
        npt.assert_allclose(result, float(jnp.pi) / 2.0, rtol=RTOL)

    def test_norm_sin(self):
        """‖sin‖₂ over [0, π] = sqrt(π/2)."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        result = float(f.norm())
        expected = math.sqrt(math.pi / 2.0)
        npt.assert_allclose(result, expected, rtol=RTOL)

    def test_norm_inf_is_vscale(self):
        """‖sin‖_∞ on [0, π] = 1."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        result = float(f.norm(p=jnp.inf))
        npt.assert_allclose(result, 1.0, rtol=1e-10)

    def test_mean_sin(self):
        """mean(sin) over [0, π] = 2/π."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        result = float(f.mean())
        expected = 2.0 / math.pi
        npt.assert_allclose(result, expected, rtol=RTOL)

    def test_diff_domain_preserved(self):
        """diff returns a Bndfun with the same domain."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        fp = f.diff()
        assert isinstance(fp, Bndfun)
        assert fp.domain == D_01PI

    def test_cumsum_domain_preserved(self):
        """cumsum returns a Bndfun with the same domain."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        F = f.cumsum()
        assert isinstance(F, Bndfun)
        assert F.domain == D_01PI


# ============================================================================
# Tier 1: Roots and extrema
# ============================================================================

class TestBndfunRoots:
    """Tests for Bndfun roots, min, max.

    NOT JIT-safe (variable output size).
    """

    def test_sin_roots_on_0_pi(self):
        """sin on (0, π) has roots only at the endpoints."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        r = f.roots()
        # sin(x) = 0 at x=0 and x=π (endpoints)
        # roots should return values in [0, π]
        assert r.shape[0] <= 2  # at most 2 boundary roots
        for ri in np.array(r):
            assert 0.0 - 1e-12 <= ri <= float(jnp.pi) + 1e-12

    def test_sin_roots_on_0_2pi(self):
        """sin on [0, 2π] has root at x=π (interior)."""
        d = Domain((0.0, 2.0 * float(jnp.pi)))
        f = Bndfun.from_function(jnp.sin, d)
        r = f.roots()
        # Check π is among the roots
        r_np = np.sort(np.array(r))
        interior = r_np[(r_np > 0.01) & (r_np < 2 * math.pi - 0.01)]
        assert len(interior) >= 1
        npt.assert_allclose(interior[0], math.pi, rtol=1e-10)

    def test_quadratic_roots(self):
        """x² - 1 on [-2, 2] has roots at ±1."""
        d = Domain((-2.0, 2.0))
        f = Bndfun.from_function(lambda x: x**2 - 1.0, d)
        r = f.roots()
        r_np = np.sort(np.array(r))
        npt.assert_allclose(r_np, np.array([-1.0, 1.0]), rtol=1e-10)

    def test_max_sin_on_0_pi(self):
        """max of sin on [0, π] is 1 at x = π/2."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        val, pos = f.max()
        npt.assert_allclose(float(val), 1.0, rtol=1e-10)
        npt.assert_allclose(float(pos), float(jnp.pi / 2), rtol=1e-10)

    def test_min_sin_on_0_pi(self):
        """min of sin on [0, π] is 0 (at endpoints)."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        val, pos = f.min()
        npt.assert_allclose(float(val), 0.0, atol=1e-12)

    def test_max_exp_on_1_3(self):
        """max of exp on [1, 3] is exp(3) at x=3."""
        f = Bndfun.from_function(jnp.exp, D_13)
        val, pos = f.max()
        npt.assert_allclose(float(val), float(jnp.exp(3.0)), rtol=1e-10)
        npt.assert_allclose(float(pos), 3.0, rtol=1e-10)

    def test_minandmax_consistent(self):
        """minandmax returns consistent (min_val, min_pos, max_val, max_pos)."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        (min_val, min_pos), (max_val, max_pos) = f.minandmax()
        npt.assert_allclose(float(f(min_pos)), float(min_val), rtol=1e-12)
        npt.assert_allclose(float(f(max_pos)), float(max_val), rtol=1e-12)

    def test_roots_in_domain(self):
        """All roots should lie in [a, b]."""
        d = Domain((-3.0, 3.0))
        f = Bndfun.from_function(lambda x: jnp.sin(x) * jnp.cos(x), d)
        r = f.roots()
        for ri in np.array(r):
            assert d.a - 1e-10 <= ri <= d.b + 1e-10


# ============================================================================
# Tier 1: Restriction
# ============================================================================

class TestBndfunRestrict:
    """Tests for Bndfun.restrict."""

    def test_restrict_full_interval(self):
        """Restricting to the full interval returns self."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = f.restrict(D_01PI.a, D_01PI.b)
        # Should be the same object or functionally identical
        xs = jnp.linspace(0, float(jnp.pi), 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(f(xs)), np.array(g(xs)), rtol=1e-12)

    def test_restrict_half_interval(self):
        """Restrict sin on [0, π] to [0, π/2]."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        half = float(jnp.pi / 2)
        g = f.restrict(0.0, half)
        assert g.domain.a == pytest.approx(0.0, abs=1e-14)
        assert g.domain.b == pytest.approx(half, abs=1e-14)
        xs = jnp.linspace(0, half, 20, dtype=jnp.float64)
        # atol=1e-14 for near-zero values at left endpoint
        npt.assert_allclose(np.array(g(xs)), np.array(jnp.sin(xs)), atol=1e-14)

    def test_restrict_integral(self):
        """∫₀^{π/2} sin dx = 1 after restriction."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = f.restrict(0.0, float(jnp.pi / 2))
        npt.assert_allclose(float(g.sum()), 1.0, rtol=RTOL)

    def test_restrict_out_of_range_raises(self):
        """Restricting beyond the domain raises ValueError."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        with pytest.raises(ValueError, match="sub-interval"):
            f.restrict(-0.5, float(jnp.pi))

    def test_restrict_returns_bndfun(self):
        """restrict returns a Bndfun."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        g = f.restrict(0.5, float(jnp.pi) - 0.5)
        assert isinstance(g, Bndfun)


# ============================================================================
# Tier 1: Simplify and Properties
# ============================================================================

class TestBndfunProperties:
    """Tests for Bndfun properties and simplify."""

    def test_n_property(self):
        """n property equals length of coefficient array."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        assert f.n == len(f.coeffs)

    def test_len(self):
        """len(f) equals f.n."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        assert len(f) == f.n

    def test_vscale_sin(self):
        """vscale of sin on [0, π] is 1."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        npt.assert_allclose(f.vscale, 1.0, rtol=1e-10)

    def test_coeffs_property(self):
        """coeffs delegates to onefun.coeffs."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        npt.assert_allclose(np.array(f.coeffs), np.array(f.onefun.coeffs))

    def test_values_property(self):
        """values at Chebyshev points match explicit evaluation."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        # The values at Chebyshev-2 points of the onefun should equal
        # f evaluated at the mapped Chebyshev points
        from chebfunjax.utils.quadrature import chebpts
        y = chebpts(f.n, kind=2)  # Chebyshev points in [-1, 1]
        x = f.domain.forward_map(y)  # Mapped to [0, π]
        # atol=1e-14 for near-zero endpoint values
        npt.assert_allclose(np.array(f.values), np.array(jnp.sin(x)), atol=1e-14)

    def test_simplify_padded(self):
        """Simplifying a padded Bndfun removes excess zeros."""
        t = Chebtech2.from_function(jnp.sin).prolong(50)
        f = Bndfun.from_chebtech(t, D_01PI)
        g = f.simplify()
        assert g.n < f.n

    def test_repr(self):
        """repr contains class name, domain, and length."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        s = repr(f)
        assert "Bndfun" in s
        assert "0" in s
        assert str(f.n) in s

    def test_ishappy(self):
        """ishappy is True after adaptive construction."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        assert f.ishappy is True


# ============================================================================
# Tier 1: JAX semantics on calculus
# ============================================================================

class TestBndfunJAXSemantics:
    """Tests that JIT-safe operations work under JAX transforms."""

    def test_jit_sum(self):
        """f.sum() can be called inside JIT."""
        f = Bndfun.from_function(jnp.sin, D_01PI)

        @jax.jit
        def compute_sum(g):
            return g.sum()

        result = compute_sum(f)
        npt.assert_allclose(float(result), 2.0, rtol=RTOL)

    def test_jit_diff_eval(self):
        """Evaluate the derivative inside JIT."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        fp = f.diff()  # computed outside JIT

        @jax.jit
        def eval_deriv(g, x):
            return g(x)

        xs = jnp.linspace(0.1, float(jnp.pi) - 0.1, 10, dtype=jnp.float64)
        ys = eval_deriv(fp, xs)
        npt.assert_allclose(np.array(ys), np.array(jnp.cos(xs)), rtol=RTOL)

    def test_grad_wrt_input(self):
        """jax.grad through Bndfun evaluation gives d/dx f(x)."""
        f = Bndfun.from_function(lambda x: x**3, D_13)
        df = jax.grad(lambda x: f(x))
        for x0 in [1.2, 1.8, 2.5]:
            x0_arr = jnp.float64(x0)
            # d/dx x³ = 3x²
            npt.assert_allclose(float(df(x0_arr)), 3 * x0**2, rtol=1e-10)

    def test_jit_inner_product(self):
        """Inner product inside JIT."""
        f = Bndfun.from_function(jnp.sin, D_01PI)

        @jax.jit
        def inner(g):
            return g.inner(g)

        result = float(inner(f))
        npt.assert_allclose(result, math.pi / 2.0, rtol=RTOL)


# ============================================================================
# Tier 2: MATLAB cross-validation
# ============================================================================

@pytest.mark.matlab
class TestBndfunMATLAB:
    """Cross-validation against MATLAB Chebfun golden references.

    These tests require tests/references/bndfun.mat to be present.
    See matlab_harness/refs/bndfun_refs.m for the generation script.

    MATLAB references verify:
    - sum (∫ sin on [0,π] = 2, ∫ exp on [1,3] = e³-e)
    - diff (derivative of sin matches cos)
    - roots (zeros of sin on [0, 2π])
    - max (max of sin on [0, π])
    """

    @pytest.fixture(autouse=True)
    def load_ref(self, request):
        from tests.conftest import load_matlab_ref
        self.ref = load_matlab_ref("bndfun.mat")

    def test_sum_sin(self):
        """∫₀^π sin matches MATLAB."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        npt.assert_allclose(float(f.sum()), float(self.ref["sum_sin_0_pi"]),
                            rtol=RTOL, atol=ATOL)

    def test_sum_exp(self):
        """∫₁^3 exp matches MATLAB."""
        f = Bndfun.from_function(jnp.exp, D_13)
        npt.assert_allclose(float(f.sum()), float(self.ref["sum_exp_1_3"]),
                            rtol=RTOL, atol=ATOL)

    def test_diff_sin_values(self):
        """Derivative of sin on [0, π] matches MATLAB at test points."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        fp = f.diff()
        xs = jnp.asarray(self.ref["test_points_0_pi"], dtype=jnp.float64)
        ys = np.array(fp(xs))
        npt.assert_allclose(ys, self.ref["diff_sin_values"], rtol=RTOL, atol=ATOL)

    def test_eval_sin_values(self):
        """Evaluation of sin on [0, π] matches MATLAB at test points."""
        f = Bndfun.from_function(jnp.sin, D_01PI)
        xs = jnp.asarray(self.ref["test_points_0_pi"], dtype=jnp.float64)
        ys = np.array(f(xs))
        npt.assert_allclose(ys, self.ref["sin_values_0_pi"], rtol=RTOL, atol=ATOL)

    def test_roots_quadratic(self):
        """Roots of x²-1 on [-2, 2] match MATLAB."""
        d = Domain((-2.0, 2.0))
        f = Bndfun.from_function(lambda x: x**2 - 1.0, d)
        r = np.sort(np.array(f.roots()))
        r_ref = np.sort(self.ref["roots_xsq_minus1"])
        npt.assert_allclose(r, r_ref, rtol=1e-10)
