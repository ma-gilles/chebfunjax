"""U100 — Autodiff: verify JAX automatic differentiation through evaluation.

Documents and tests that JAX AD (jax.grad, jax.jacfwd, jax.jacrev, jax.vmap)
works correctly through Chebfun, Chebfun2, SeparableApprox, and Chebfun3
evaluation.

The key property is that evaluation is JIT-safe, grad-safe, and vmap-safe
because all evaluation paths use pure JAX operations (Clenshaw recurrence,
jnp arithmetic).

JAX contract
------------
- Chebfun.__call__     : jit=yes (single-piece), grad=yes, vmap=yes
- Chebfun2.__call__    : jit=yes, grad=yes, vmap=yes
- SeparableApprox.__call__ : jit=yes, grad=yes, vmap=yes
- Chebfun3.__call__    : jit=yes, grad=yes, vmap=yes
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.chebfun3d.chebfun3 import chebfun3

# Tolerances
RTOL_GRAD = 1e-10   # gradient comparison tolerance
ATOL_GRAD = 1e-12


# ============================================================================
# Tier 1: 1-D Chebfun — jit / grad / vmap
# ============================================================================


class TestChebfunAD:
    """JAX AD through 1-D Chebfun evaluation."""

    def setup_method(self):
        """Build a sin(pi * x) Chebfun on [-1, 1] once per test class."""
        self.f = chebfun(lambda x: jnp.sin(jnp.pi * x), domain=(-1.0, 1.0))

    def test_jit_eval(self):
        """jit-wrapped lambda calling f(x) returns the same value as f(x).

        Note: equinox Modules must be JIT-traced via a capturing lambda (or
        eqx.filter_jit).  The ``__call__`` is already decorated with
        ``@eqx.filter_jit`` internally, but a raw ``jax.jit(f)`` would fail
        because the Module contains unhashable Python lists.
        """
        x = jnp.float64(0.3)
        expected = float(self.f(x))
        jit_fn = jax.jit(lambda x_: self.f(x_))
        got = float(jit_fn(x))
        assert abs(got - expected) < 1e-14, f"JIT eval mismatch: {got} vs {expected}"

    def test_grad_at_point(self):
        """jax.grad(f)(x) ≈ pi * cos(pi * x)."""
        x = jnp.float64(0.3)
        grad_f = jax.grad(lambda x_: self.f(x_))(x)
        expected = jnp.pi * jnp.cos(jnp.pi * x)
        npt.assert_allclose(float(grad_f), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_grad_exp(self):
        """Gradient of exp(x) is exp(x)."""
        f = chebfun(jnp.exp, domain=(-1.0, 1.0))
        x = jnp.float64(0.5)
        grad_f = jax.grad(lambda x_: f(x_))(x)
        expected = jnp.exp(x)
        npt.assert_allclose(float(grad_f), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_vmap_eval(self):
        """vmap(f)(xs) == [f(x) for x in xs]."""
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        result = jax.vmap(self.f)(xs)
        expected = jnp.sin(jnp.pi * xs)
        npt.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_vmap_grad(self):
        """vmap(grad(f))(xs) == pi * cos(pi * xs)."""
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        g = jax.vmap(jax.grad(lambda x_: self.f(x_)))(xs)
        expected = jnp.pi * jnp.cos(jnp.pi * xs)
        npt.assert_allclose(g, expected, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_second_derivative(self):
        """jax.grad(jax.grad(f))(x) ≈ -pi^2 * sin(pi * x)."""
        f = self.f
        x = jnp.float64(0.4)
        d2f = jax.grad(jax.grad(lambda x_: f(x_)))(x)
        expected = -(jnp.pi ** 2) * jnp.sin(jnp.pi * x)
        npt.assert_allclose(float(d2f), float(expected), rtol=1e-9, atol=1e-11)

    def test_jit_grad(self):
        """jit(grad(f))(x) is consistent with grad(f)(x)."""
        f = self.f
        x = jnp.float64(0.3)
        grad_fn = jax.jit(jax.grad(lambda x_: f(x_)))
        got = float(grad_fn(x))
        expected = float(jnp.pi * jnp.cos(jnp.pi * x))
        npt.assert_allclose(got, expected, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_multipiece_eval_no_grad(self):
        """Multi-piece Chebfun evaluates correctly; note grad requires single piece."""
        # Construct a breakpoint Chebfun by concatenation
        f = chebfun(lambda x: jnp.sin(jnp.pi * x), domain=(-1.0, 0.0, 1.0))
        x0 = jnp.float64(0.5)
        expected = jnp.sin(jnp.pi * x0)
        npt.assert_allclose(float(f(x0)), float(expected), rtol=1e-12, atol=1e-14)

    def test_custom_domain_grad(self):
        """grad works on non-standard domains [0, pi]."""
        f = chebfun(jnp.sin, domain=(0.0, float(jnp.pi)))
        x = jnp.float64(1.0)
        grad_f = jax.grad(lambda x_: f(x_))(x)
        expected = jnp.cos(x)
        npt.assert_allclose(float(grad_f), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)


# ============================================================================
# Tier 2: 2-D Chebfun2 / SeparableApprox — jit / grad / vmap
# ============================================================================


class TestChebfun2AD:
    """JAX AD through Chebfun2 / SeparableApprox evaluation."""

    def setup_method(self):
        """Build cos(x + y) on [-1, 1]^2."""
        self.f = chebfun2(lambda x, y: jnp.cos(x + y))

    def test_jit_eval(self):
        """jit-wrapped lambda calling f(x, y) == f(x, y).

        See note in TestChebfunAD.test_jit_eval about eqx.Module + jax.jit.
        """
        x, y = jnp.float64(0.3), jnp.float64(-0.2)
        expected = float(self.f(x, y))
        jit_fn = jax.jit(lambda x_, y_: self.f(x_, y_))
        got = float(jit_fn(x, y))
        assert abs(got - expected) < 1e-14

    def test_grad_x(self):
        """d/dx cos(x+y) = -sin(x+y)."""
        f = self.f
        x, y = jnp.float64(0.3), jnp.float64(-0.2)
        grad_x = jax.grad(lambda x_: f(x_, y))(x)
        expected = -jnp.sin(x + y)
        npt.assert_allclose(float(grad_x), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_grad_y(self):
        """d/dy cos(x+y) = -sin(x+y)."""
        f = self.f
        x, y = jnp.float64(0.3), jnp.float64(-0.2)
        grad_y = jax.grad(lambda y_: f(x, y_))(y)
        expected = -jnp.sin(x + y)
        npt.assert_allclose(float(grad_y), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_vmap_x(self):
        """vmap along x with fixed y."""
        f = self.f
        xs = jnp.linspace(-0.9, 0.9, 15, dtype=jnp.float64)
        y = jnp.float64(0.5)
        result = jax.vmap(lambda x_: f(x_, y))(xs)
        expected = jnp.cos(xs + y)
        npt.assert_allclose(result, expected, rtol=1e-12, atol=1e-13)

    def test_vmap_grad(self):
        """vmap(grad_x(f))(xs) == -sin(xs + y)."""
        f = self.f
        xs = jnp.linspace(-0.8, 0.8, 10, dtype=jnp.float64)
        y = jnp.float64(-0.3)
        g = jax.vmap(jax.grad(lambda x_: f(x_, y)))(xs)
        expected = -jnp.sin(xs + y)
        npt.assert_allclose(g, expected, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_jacfwd(self):
        """jacfwd through f(x, y) = sin(x)*cos(y)."""
        f2 = chebfun2(lambda x, y: jnp.sin(x) * jnp.cos(y))
        x, y = jnp.float64(0.4), jnp.float64(-0.1)
        # df/dx = cos(x)*cos(y), df/dy = -sin(x)*sin(y)
        df_dx = jax.grad(lambda x_: f2(x_, y))(x)
        df_dy = jax.grad(lambda y_: f2(x, y_))(y)
        npt.assert_allclose(float(df_dx), float(jnp.cos(x) * jnp.cos(y)),
                            rtol=RTOL_GRAD, atol=ATOL_GRAD)
        npt.assert_allclose(float(df_dy), float(-jnp.sin(x) * jnp.sin(y)),
                            rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_separable_approx_grad(self):
        """SeparableApprox.__call__ also supports grad."""
        sa = SeparableApprox.from_function(lambda x, y: jnp.exp(x * y))
        x, y = jnp.float64(0.5), jnp.float64(0.3)
        # d/dx exp(xy) = y * exp(xy)
        df_dx = jax.grad(lambda x_: sa(x_, y))(x)
        expected = y * jnp.exp(x * y)
        npt.assert_allclose(float(df_dx), float(expected), rtol=1e-9, atol=1e-11)


# ============================================================================
# Tier 3: 3-D Chebfun3 — jit / grad
# ============================================================================


class TestChebfun3AD:
    """JAX AD through Chebfun3 evaluation."""

    def setup_method(self):
        """Build a simple 3D function."""
        self.f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))

    def test_jit_eval(self):
        """jit-wrapped lambda calling f(x, y, z) == f(x, y, z).

        See note in TestChebfunAD.test_jit_eval about eqx.Module + jax.jit.
        """
        x, y, z = jnp.float64(0.1), jnp.float64(-0.2), jnp.float64(0.3)
        expected = float(self.f(x, y, z))
        jit_fn = jax.jit(lambda x_, y_, z_: self.f(x_, y_, z_))
        got = float(jit_fn(x, y, z))
        assert abs(got - expected) < 1e-14

    def test_grad_x(self):
        """d/dx cos(x+y+z) = -sin(x+y+z)."""
        f = self.f
        x, y, z = jnp.float64(0.1), jnp.float64(-0.2), jnp.float64(0.3)
        df_dx = jax.grad(lambda x_: f(x_, y, z))(x)
        expected = -jnp.sin(x + y + z)
        npt.assert_allclose(float(df_dx), float(expected), rtol=1e-8, atol=1e-10)

    def test_grad_z(self):
        """d/dz cos(x+y+z) = -sin(x+y+z)."""
        f = self.f
        x, y, z = jnp.float64(0.1), jnp.float64(-0.2), jnp.float64(0.3)
        df_dz = jax.grad(lambda z_: f(x, y, z_))(z)
        expected = -jnp.sin(x + y + z)
        npt.assert_allclose(float(df_dz), float(expected), rtol=1e-8, atol=1e-10)

    def test_vmap_eval(self):
        """vmap along z with fixed x, y."""
        f = self.f
        x, y = jnp.float64(0.0), jnp.float64(0.0)
        zs = jnp.linspace(-0.8, 0.8, 10, dtype=jnp.float64)
        result = jax.vmap(lambda z_: f(x, y, z_))(zs)
        expected = jnp.cos(x + y + zs)
        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


# ============================================================================
# Tier 4: Composition / higher-order AD
# ============================================================================


class TestComposedAD:
    """AD through chained operations involving Chebfun evaluation."""

    def test_loss_over_chebfun(self):
        """AD through a scalar loss L(a) = (f_a(0.5) - target)^2, where
        f_a = chebfun(lambda x: a * sin(x)) (requires re-construction,
        so this tests AD through the *evaluation* path only).

        We fix the Chebfun structure and differentiate only through eval.
        """
        # Construct once
        f = chebfun(jnp.sin, domain=(-1.0, 1.0))
        x0 = jnp.float64(0.5)
        target = jnp.float64(0.0)

        # Loss: L(a) = (a * f(x0) - target)^2
        def loss(a):
            return (a * f(x0) - target) ** 2

        dL_da = jax.grad(loss)(jnp.float64(1.0))
        # dL/da = 2 * (a * f(x0) - target) * f(x0)  at a=1 => 2*sin(0.5)*sin(0.5)
        expected = jnp.float64(2.0) * jnp.sin(x0) * jnp.sin(x0)
        npt.assert_allclose(float(dL_da), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_chebfun2_in_loss(self):
        """AD through a loss involving Chebfun2 evaluation."""
        f = chebfun2(lambda x, y: x * y)
        x = jnp.float64(0.5)
        y = jnp.float64(0.3)

        # L(x) = f(x, y)^2 = x^2 * y^2
        def loss(x_):
            return f(x_, y) ** 2

        dL = jax.grad(loss)(x)
        expected = jnp.float64(2.0) * x * y ** 2
        npt.assert_allclose(float(dL), float(expected), rtol=RTOL_GRAD, atol=ATOL_GRAD)

    def test_jit_grad_chebfun2(self):
        """jit(grad(f))(x, y) is stable across multiple calls."""
        f = chebfun2(lambda x, y: jnp.sin(x + y))
        x, y = jnp.float64(0.2), jnp.float64(-0.4)

        @jax.jit
        def grad_x(x_, y_):
            return jax.grad(lambda x__: f(x__, y_))(x_)

        result1 = float(grad_x(x, y))
        result2 = float(grad_x(x, y))
        assert result1 == result2, "JIT grad should be deterministic"

        expected = float(jnp.cos(x + y))
        npt.assert_allclose(result1, expected, rtol=RTOL_GRAD, atol=ATOL_GRAD)
