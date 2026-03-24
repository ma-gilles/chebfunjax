"""Tests for chebfunjax.fun.unbndfun — Unbndfun.

Tests the unbounded-domain function representation.

JAX contract:
- __call__: jit=YES, vmap=YES, grad=YES
- sum, inner, norm, diff, cumsum: construction NOT JIT-safe, evaluation YES
- Arithmetic operators: result is JIT-safe for evaluation

Test domains:
- (-∞, ∞)   — exp(-x²),   ∫ = √π, classic Gaussian
- [0, ∞)    — exp(-x),    ∫ = 1
- (-∞, 0]   — exp(x),     ∫ = 1
- [1, ∞)    — 1/(1+x²),   ∫ = π/4 (improper integral)

MATLAB golden references are computed analytically for the mappings:
  Right semi-infinite:   x = 15*(y+1)/(1-y) + a,  dx/dy = 30/(y-1)²
  Left  semi-infinite:   x = 15*(y-1)/(y+1) + b,  dx/dy = 30/(y+1)²
  Doubly infinite:       x = 5*y/(1-y²),           dx/dy = 5*(1+y²)/(1-y²)²
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import (
    Unbndfun,
    _derivative_both,
    _derivative_left,
    _derivative_right,
    _forward_both,
    _forward_left,
    _forward_right,
    _inverse_both,
    _inverse_left,
    _inverse_right,
    _mapping_type,
    _validate_unbounded_domain,
)
from chebfunjax.tech.chebtech import Chebtech2

# Tolerances
RTOL = 1e-10
ATOL = 1e-12


# ============================================================================
# Tier 0: Mapping helpers
# ============================================================================


class TestMappingHelpers:
    """Tests for the pure-function mapping helpers.

    JAX contract: jit=YES, vmap=YES, grad=YES
    """

    def test_forward_inverse_right(self):
        """forward_right and inverse_right are inverse to each other."""
        y = jnp.linspace(-0.9, 0.9, 50, dtype=jnp.float64)
        a = 2.0
        x = _forward_right(y, a)
        y2 = _inverse_right(x, a)
        npt.assert_allclose(np.array(y2), np.array(y), rtol=1e-14)

    def test_forward_right_endpoints(self):
        """forward_right maps -1 → a, and diverges as y→1."""
        a = 3.0
        x_left = float(_forward_right(jnp.float64(-1.0), a))
        assert abs(x_left - a) < 1e-14, f"Expected {a}, got {x_left}"
        # y=0.99 should give a large positive number
        x_near_inf = float(_forward_right(jnp.float64(0.99), a))
        assert x_near_inf > 100.0

    def test_forward_inverse_left(self):
        """forward_left and inverse_left are inverse to each other."""
        y = jnp.linspace(-0.9, 0.9, 50, dtype=jnp.float64)
        b = -1.0
        x = _forward_left(y, b)
        y2 = _inverse_left(x, b)
        npt.assert_allclose(np.array(y2), np.array(y), rtol=1e-14)

    def test_forward_left_endpoints(self):
        """forward_left maps 1 → b, and diverges as y→-1."""
        b = -2.0
        x_right = float(_forward_left(jnp.float64(1.0), b))
        assert abs(x_right - b) < 1e-14
        x_near_ninf = float(_forward_left(jnp.float64(-0.99), b))
        assert x_near_ninf < -100.0

    def test_forward_inverse_both(self):
        """forward_both and inverse_both are inverse to each other."""
        y = jnp.linspace(-0.9, 0.9, 50, dtype=jnp.float64)
        x = _forward_both(y)
        y2 = _inverse_both(x)
        npt.assert_allclose(np.array(y2), np.array(y), rtol=1e-14)

    def test_forward_both_at_zero(self):
        """forward_both maps 0 → 0."""
        x = float(_forward_both(jnp.float64(0.0)))
        assert abs(x) < 1e-15

    def test_derivative_right_positive(self):
        """derivative_right is positive on (-1, 1)."""
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        der = _derivative_right(y)
        assert jnp.all(der > 0)

    def test_derivative_left_positive(self):
        """derivative_left is positive on (-1, 1)."""
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        der = _derivative_left(y)
        assert jnp.all(der > 0)

    def test_derivative_both_positive(self):
        """derivative_both is positive on (-1, 1)."""
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        der = _derivative_both(y)
        assert jnp.all(der > 0)

    def test_derivative_right_value_at_zero(self):
        """derivative_right at y=0 equals 30/(0-1)^2 = 30."""
        der = float(_derivative_right(jnp.float64(0.0)))
        npt.assert_allclose(der, 30.0, rtol=1e-15)

    def test_derivative_left_value_at_zero(self):
        """derivative_left at y=0 equals 30/(0+1)^2 = 30."""
        der = float(_derivative_left(jnp.float64(0.0)))
        npt.assert_allclose(der, 30.0, rtol=1e-15)

    def test_derivative_both_value_at_zero(self):
        """derivative_both at y=0 equals 5*(1+0)/(1-0)^2 = 5."""
        der = float(_derivative_both(jnp.float64(0.0)))
        npt.assert_allclose(der, 5.0, rtol=1e-15)

    def test_jit_forward_right(self):
        """JIT: forward_right is JIT-safe."""
        import functools
        f = jax.jit(functools.partial(_forward_right, a=0.0))
        y = jnp.linspace(-0.5, 0.5, 10, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(f(y)),
            np.array(_forward_right(y, 0.0)),
            rtol=1e-15,
        )

    def test_jit_inverse_both(self):
        """JIT: inverse_both is JIT-safe."""
        f = jax.jit(_inverse_both)
        x = jnp.linspace(-10.0, 10.0, 20, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(f(x)),
            np.array(_inverse_both(x)),
            rtol=1e-15,
        )

    def test_mapping_type_dispatch(self):
        """_mapping_type returns correct strings."""
        assert _mapping_type(Domain((0.0, jnp.inf))) == "right_inf"
        assert _mapping_type(Domain((-jnp.inf, 0.0))) == "left_inf"
        assert _mapping_type(Domain((-jnp.inf, jnp.inf))) == "both_inf"


# ============================================================================
# Tier 1: Construction
# ============================================================================


class TestUnbndfunConstruction:
    """Tests for Unbndfun construction.

    JAX contract: construction NOT JIT-safe (adaptive)
    """

    def test_from_function_right_inf(self):
        """Construct on [0, ∞) from exp(-x)."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        assert f.ishappy
        assert f.n > 1
        assert f.mapping_type == "right_inf"

    def test_from_function_left_inf(self):
        """Construct on (-∞, 0] from exp(x)."""
        d = Domain((-jnp.inf, 0.0))
        f = Unbndfun.from_function(jnp.exp, d)
        assert f.ishappy
        assert f.mapping_type == "left_inf"

    def test_from_function_both_inf(self):
        """Construct on (-∞, ∞) from exp(-x²)."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        assert f.ishappy
        assert f.mapping_type == "both_inf"

    def test_from_function_fixed_n(self):
        """Fixed-n construction sets exact length."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d, n=30)
        assert f.n == 30

    def test_from_chebtech(self):
        """from_chebtech wraps an existing Chebtech2."""
        d = Domain((0.0, jnp.inf))
        t = Chebtech2.from_function(lambda y: jnp.exp(-(15.0 * (y + 1.0) / (1.0 - y))))
        f = Unbndfun.from_chebtech(t, d)
        assert f.onefun is t
        assert f.domain == d
        assert f.mapping_type == "right_inf"

    def test_invalid_bounded_domain(self):
        """from_function raises ValueError for a bounded domain."""
        d = Domain((0.0, 1.0))
        with pytest.raises(ValueError, match="at least one infinite"):
            Unbndfun.from_function(jnp.sin, d)

    def test_invalid_multi_interval_domain(self):
        """from_function raises ValueError for piecewise domains."""
        d = Domain((-jnp.inf, 0.0, jnp.inf))
        with pytest.raises(ValueError, match="single-interval"):
            Unbndfun.from_function(jnp.sin, d)

    def test_non_default_left_endpoint(self):
        """Construct on [2, ∞)."""
        d = Domain((2.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: 1.0 / (1.0 + (x - 2.0) ** 2), d)
        assert f.ishappy
        assert f.mapping_type == "right_inf"

    def test_non_default_right_endpoint(self):
        """Construct on (-∞, -1]."""
        d = Domain((-jnp.inf, -1.0))
        f = Unbndfun.from_function(lambda x: jnp.exp(x), d)
        assert f.ishappy
        assert f.mapping_type == "left_inf"

    def test_domain_attribute(self):
        """domain attribute matches construction domain."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        assert f.domain == d


# ============================================================================
# Tier 2: Evaluation
# ============================================================================


class TestUnbndfunEval:
    """Tests for Unbndfun evaluation (__call__).

    JAX contract: jit=YES, vmap=YES, grad=YES
    """

    def test_eval_right_inf_interior(self):
        """Evaluate exp(-x) on [0, ∞) at interior points."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        xs = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)
        expected = np.exp(-np.array([0.0, 1.0, 2.0, 5.0, 10.0]))
        npt.assert_allclose(np.array(f(xs)), expected, rtol=1e-11, atol=1e-14)

    def test_eval_left_inf_interior(self):
        """Evaluate exp(x) on (-∞, 0] at interior points."""
        d = Domain((-jnp.inf, 0.0))
        f = Unbndfun.from_function(jnp.exp, d)
        xs = jnp.array([-10.0, -5.0, -2.0, -1.0, 0.0], dtype=jnp.float64)
        expected = np.exp(np.array([-10.0, -5.0, -2.0, -1.0, 0.0]))
        npt.assert_allclose(np.array(f(xs)), expected, rtol=1e-12)

    def test_eval_both_inf_interior(self):
        """Evaluate exp(-x²) on (-∞, ∞) at interior points."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        xs = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=jnp.float64)
        expected = np.exp(-np.array([-3.0, -1.0, 0.0, 1.0, 3.0]) ** 2)
        npt.assert_allclose(np.array(f(xs)), expected, rtol=1e-12)

    def test_eval_at_finite_endpoint(self):
        """Evaluate at the finite endpoint of [0, ∞) domain."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        val = float(f(jnp.float64(0.0)))
        npt.assert_allclose(val, 1.0, rtol=1e-12)

    def test_eval_at_inf(self):
        """Evaluating at +∞ gives the value at y=1 (boundary of onefun)."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        # exp(-∞) = 0; onefun(1) should be ~0
        val = float(f(jnp.float64(jnp.inf)))
        assert abs(val) < 1e-10

    def test_jit_eval(self):
        """__call__ is JIT-safe."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        # Use a lambda closure over f, not jax.jit(f) directly, because
        # equinox modules with array fields are not directly hashable.
        f_jit = jax.jit(lambda x: f(x))
        xs = jnp.linspace(-2.0, 2.0, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(f_jit(xs)), np.array(f(xs)), rtol=1e-14)

    def test_vmap_eval(self):
        """__call__ is vmap-safe."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        xs = jnp.linspace(-3.0, 3.0, 30, dtype=jnp.float64)
        # vmap over scalar inputs
        f_vmap = jax.vmap(lambda x: f(x))
        npt.assert_allclose(np.array(f_vmap(xs)), np.array(f(xs)), rtol=1e-14)

    def test_grad_eval(self):
        """__call__ is grad-safe: d/dx[exp(-x²)] at x=1 is -2*exp(-1)."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        grad_f = jax.grad(lambda x: f(x))
        result = float(grad_f(jnp.float64(1.0)))
        expected = -2.0 * math.exp(-1.0)
        npt.assert_allclose(result, expected, rtol=1e-9)

    def test_scalar_eval(self):
        """Scalar evaluation returns a scalar."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        val = f(jnp.float64(1.0))
        assert val.shape == ()


# ============================================================================
# Tier 3: Integration (definite)
# ============================================================================


class TestUnbndfunSum:
    """Tests for Unbndfun.sum().

    Key identities:
    - ∫_{-∞}^{∞} exp(-x²) dx = √π
    - ∫_0^{∞} exp(-x) dx = 1
    - ∫_{-∞}^0 exp(x) dx = 1
    - ∫_0^{∞} x*exp(-x²) dx = 1/2   (Gaussian half-moment)
    - ∫_{-∞}^{∞} 1/(1+x²) dx = π
    """

    def test_gaussian_integral(self):
        """∫_{-∞}^{∞} exp(-x²) dx = √π."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        result = float(f.sum())
        npt.assert_allclose(result, math.sqrt(math.pi), rtol=1e-10)

    def test_exponential_decay_right(self):
        """∫_0^{∞} exp(-x) dx = 1."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        result = float(f.sum())
        npt.assert_allclose(result, 1.0, rtol=1e-10)

    def test_exponential_decay_left(self):
        """∫_{-∞}^0 exp(x) dx = 1."""
        d = Domain((-jnp.inf, 0.0))
        f = Unbndfun.from_function(jnp.exp, d)
        result = float(f.sum())
        npt.assert_allclose(result, 1.0, rtol=1e-10)

    def test_lorentzian_integral(self):
        """∫_{-∞}^{∞} 1/(1+x²) dx = π."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: 1.0 / (1.0 + x ** 2), d)
        result = float(f.sum())
        npt.assert_allclose(result, math.pi, rtol=1e-10)

    def test_gaussian_half_moment(self):
        """∫_0^{∞} x*exp(-x²) dx = 1/2."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: x * jnp.exp(-(x ** 2)), d)
        result = float(f.sum())
        npt.assert_allclose(result, 0.5, rtol=1e-10)

    def test_shifted_right_domain(self):
        """∫_1^{∞} exp(-x) dx = exp(-1)."""
        d = Domain((1.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        result = float(f.sum())
        npt.assert_allclose(result, math.exp(-1.0), rtol=1e-10)

    def test_shifted_left_domain(self):
        """∫_{-∞}^{-1} exp(x) dx = exp(-1)."""
        d = Domain((-jnp.inf, -1.0))
        f = Unbndfun.from_function(jnp.exp, d)
        result = float(f.sum())
        npt.assert_allclose(result, math.exp(-1.0), rtol=1e-10)


# ============================================================================
# Tier 4: Differentiation
# ============================================================================


class TestUnbndfunDiff:
    """Tests for Unbndfun.diff().

    Key identities:
    - d/dx[exp(-x)] = -exp(-x)  on [0, ∞)
    - d/dx[exp(-x²)] = -2x*exp(-x²)  on (-∞, ∞)
    """

    def test_diff_exp_decay_right(self):
        """d/dx[exp(-x)] = -exp(-x) on [0, ∞)."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        df = f.diff()
        xs = jnp.array([0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        expected = -np.exp(-np.array([0.5, 1.0, 2.0, 5.0]))
        npt.assert_allclose(np.array(df(xs)), expected, rtol=1e-9)

    def test_diff_gaussian_both_inf(self):
        """d/dx[exp(-x²)] = -2x*exp(-x²) on (-∞, ∞)."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        df = f.diff()
        xs = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float64)
        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = -2.0 * x_np * np.exp(-(x_np ** 2))
        # atol=1e-14 to handle machine-epsilon noise near zero at x=0
        npt.assert_allclose(np.array(df(xs)), expected, rtol=1e-9, atol=1e-14)

    def test_diff_order_zero(self):
        """diff(k=0) returns the function itself."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        f0 = f.diff(k=0)
        xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(f0(xs)), np.array(f(xs)), rtol=1e-14)

    def test_diff_preserves_domain(self):
        """diff() result has the same domain as the original."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        df = f.diff()
        assert df.domain == d
        assert df.mapping_type == f.mapping_type


# ============================================================================
# Tier 5: Indefinite integral (cumsum)
# ============================================================================


class TestUnbndfunCumsum:
    """Tests for Unbndfun.cumsum().

    The antiderivative F satisfies F(a) = 0 for the finite endpoint a.
    """

    def test_cumsum_exp_decay_right(self):
        """Antiderivative of exp(-x) on [0, ∞): F(x) = 1 - exp(-x)."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        F = f.cumsum()
        xs = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64)
        expected = 1.0 - np.exp(-np.array([0.0, 1.0, 2.0, 3.0]))
        npt.assert_allclose(np.array(F(xs)), expected, rtol=1e-10)

    def test_cumsum_preserves_domain(self):
        """cumsum() result has the same domain as the original."""
        d = Domain((-jnp.inf, 0.0))
        f = Unbndfun.from_function(jnp.exp, d)
        F = f.cumsum()
        assert F.domain == d
        assert F.mapping_type == f.mapping_type


# ============================================================================
# Tier 6: Arithmetic
# ============================================================================


class TestUnbndfunArithmetic:
    """Tests for Unbndfun arithmetic operators.

    JAX contract: arithmetic on Unbndfuns returns new Unbndfuns; evaluation
    of the result is JIT-safe.
    """

    def test_add(self):
        """f + g evaluates correctly."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        g = Unbndfun.from_function(lambda x: jnp.exp(-2.0 * x ** 2), d)
        h = f + g
        xs = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float64)
        x_np = np.array([-1.0, 0.0, 1.0])
        expected = np.exp(-(x_np ** 2)) + np.exp(-2.0 * x_np ** 2)
        npt.assert_allclose(np.array(h(xs)), expected, rtol=1e-12)

    def test_sub(self):
        """f - g evaluates correctly."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        g = Unbndfun.from_function(lambda x: 0.5 * jnp.exp(-x), d)
        h = f - g
        xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(h(xs)),
            0.5 * np.exp(-np.array([1.0, 2.0, 3.0])),
            rtol=1e-12,
        )

    def test_scalar_mul(self):
        """f * scalar evaluates correctly."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        h = 3.0 * f
        xs = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(h(xs)),
            3.0 * np.exp(-np.array([0.0, 1.0, 2.0])),
            rtol=1e-12,
        )

    def test_mul(self):
        """f * g evaluates correctly."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        g = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        h = f * g
        xs = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float64)
        x_np = np.array([-1.0, 0.0, 1.0])
        expected = np.exp(-2.0 * x_np ** 2)
        npt.assert_allclose(np.array(h(xs)), expected, rtol=1e-12)

    def test_neg(self):
        """-f evaluates correctly."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        h = -f
        xs = jnp.array([1.0, 2.0], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(h(xs)),
            -np.exp(-np.array([1.0, 2.0])),
            rtol=1e-12,
        )

    def test_arithmetic_domain_mismatch(self):
        """Arithmetic on mismatched domains raises ValueError."""
        d1 = Domain((0.0, jnp.inf))
        d2 = Domain((1.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d1)
        g = Unbndfun.from_function(lambda x: jnp.exp(-x), d2)
        with pytest.raises(ValueError, match="domains do not match"):
            _ = f + g

    def test_sum_after_arithmetic(self):
        """Integral of 2*exp(-x) on [0,∞) = 2."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        h = 2.0 * f
        result = float(h.sum())
        npt.assert_allclose(result, 2.0, rtol=1e-10)


# ============================================================================
# Tier 7: norm and inner product
# ============================================================================


class TestUnbndfunNorm:
    """Tests for Unbndfun.norm() and inner()."""

    def test_l2_norm_gaussian(self):
        """||exp(-x²)||_2 = (π/2)^{1/4} on (-∞,∞)."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        result = float(f.norm(2.0))
        expected = (math.pi / 2.0) ** 0.25
        npt.assert_allclose(result, expected, rtol=1e-8)

    def test_linf_norm(self):
        """sup-norm of exp(-x²) on (-∞,∞) is 1.0 (attained at x=0)."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        result = float(f.norm(jnp.inf))
        npt.assert_allclose(result, 1.0, rtol=1e-12)

    def test_inner_product_self(self):
        """<f, f> = ||f||_2^2."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        inner = float(f.inner(f))
        norm_sq = float(f.norm(2.0)) ** 2
        npt.assert_allclose(inner, norm_sq, rtol=1e-9)

    def test_inner_orthogonal(self):
        """<exp(-x), exp(-3x)> on [0,∞) = ∫_0^∞ exp(-4x) dx = 1/4."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        g = Unbndfun.from_function(lambda x: jnp.exp(-3.0 * x), d)
        result = float(f.inner(g))
        npt.assert_allclose(result, 0.25, rtol=1e-9)


# ============================================================================
# Tier 8: Representation / display
# ============================================================================


class TestUnbndfunRepr:
    """Tests for Unbndfun.__repr__."""

    def test_repr_both_inf(self):
        """repr contains 'Unbndfun' and '-inf' and 'inf'."""
        d = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-(x ** 2)), d)
        r = repr(f)
        assert "Unbndfun" in r
        assert "-inf" in r
        assert "inf" in r

    def test_repr_right_inf(self):
        """repr for [0, ∞) contains '0' and 'inf'."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        r = repr(f)
        assert "Unbndfun" in r
        assert "inf" in r

    def test_props_accessible(self):
        """n, coeffs, values, vscale, ishappy are accessible."""
        d = Domain((0.0, jnp.inf))
        f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        assert f.n == len(f.coeffs)
        assert f.values.shape == (f.n,)
        assert f.vscale >= 0.0
        assert f.ishappy


# ============================================================================
# Tier 9: Validation
# ============================================================================


class TestUnbndfunValidation:
    """Tests for domain validation."""

    def test_reject_bounded(self):
        """Bounded domain raises ValueError."""
        d = Domain((0.0, 1.0))
        with pytest.raises(ValueError, match="at least one infinite"):
            _validate_unbounded_domain(d)

    def test_reject_multi_interval(self):
        """Multi-interval domain raises ValueError."""
        d = Domain((-jnp.inf, 0.0, jnp.inf))
        with pytest.raises(ValueError, match="single-interval"):
            _validate_unbounded_domain(d)
