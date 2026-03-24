"""Tests for SeparableApprox — low-rank 2D function approximation.

JAX contract:
    construction   : jit=NO (Python adaptive loop)
    evaluation     : jit=YES, vmap=YES, grad=YES
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.chebfun2d.separable_approx import SeparableApprox

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
RTOL = 1e-10   # evaluation accuracy
RTOL_TIGHT = 1e-12  # for exact low-rank functions


# ===========================================================================
# Class TestConstruction — from_function, rank, domain
# ===========================================================================


class TestConstruction:
    """Tests for SeparableApprox.from_function.

    JAX contract: construction is jit=NO (Python adaptive loop).
    """

    def test_cos_x_plus_y_rank(self):
        """cos(x+y) has rank exactly 2 (sum-of-products trig identity).

        cos(x+y) = cos(x)cos(y) - sin(x)sin(y) — a rank-2 function.
        """
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))
        assert f.rank == 2, (
            f"cos(x+y) should be rank 2, got rank {f.rank}"
        )

    def test_exp_xy_low_rank(self):
        """exp(x*y) should converge to a low rank approximation."""
        f = SeparableApprox.from_function(lambda x, y: jnp.exp(x * y))
        assert f.rank >= 1
        assert f.rank <= 30, f"exp(x*y) rank {f.rank} is unexpectedly high"

    def test_polynomial_xy_rank(self):
        """x*y is exactly rank-1."""
        f = SeparableApprox.from_function(lambda x, y: x * y)
        assert f.rank == 1, f"x*y should be rank 1, got rank {f.rank}"

    def test_constant_rank(self):
        """A constant function is rank 1."""
        f = SeparableApprox.from_function(lambda x, y: jnp.ones_like(x) * 3.14)
        assert f.rank == 1

    def test_domain_stored(self):
        """Domain is stored correctly."""
        f = SeparableApprox.from_function(
            lambda x, y: jnp.sin(x) * jnp.cos(y),
            domain=(0.0, 1.0, -2.0, 2.0),
        )
        assert f.domain == (0.0, 1.0, -2.0, 2.0)

    def test_default_domain(self):
        """Default domain is (-1, 1, -1, 1)."""
        f = SeparableApprox.from_function(lambda x, y: x + y)
        assert f.domain == (-1.0, 1.0, -1.0, 1.0)

    def test_repr(self):
        """repr includes rank and domain."""
        f = SeparableApprox.from_function(lambda x, y: x * y)
        s = repr(f)
        assert "SeparableApprox" in s
        assert "rank=" in s
        assert "domain=" in s

    def test_pivots_shape(self):
        """pivots array has shape (rank,)."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))
        assert f.pivots.shape == (f.rank,)

    def test_cols_rows_length(self):
        """cols and rows lists have length == rank."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))
        assert len(f.cols) == f.rank
        assert len(f.rows) == f.rank


# ===========================================================================
# Class TestEvaluation — __call__, pointwise accuracy
# ===========================================================================


class TestEvaluation:
    """Tests for SeparableApprox.__call__.

    JAX contract: evaluation is jit=YES, vmap=YES, grad=YES.
    """

    def test_cos_xy_pointwise(self):
        """cos(x+y) evaluation matches direct computation."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))
        x_test = jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        xx = xx.ravel()
        yy = yy.ravel()
        got = f(xx, yy)
        expected = jnp.cos(xx + yy)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL_TIGHT)

    def test_exp_xy_pointwise(self):
        """exp(x*y) evaluation matches direct computation."""
        f = SeparableApprox.from_function(lambda x, y: jnp.exp(x * y))
        x_test = jnp.linspace(-1.0, 1.0, 15, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 15, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        xx = xx.ravel()
        yy = yy.ravel()
        got = f(xx, yy)
        expected = jnp.exp(xx * yy)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_sin_x_cos_y_pointwise(self):
        """sin(x)*cos(y) is rank-1; evaluation matches direct."""
        f = SeparableApprox.from_function(lambda x, y: jnp.sin(x) * jnp.cos(y))
        assert f.rank == 1
        x_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        xx = xx.ravel()
        yy = yy.ravel()
        got = f(xx, yy)
        expected = jnp.sin(xx) * jnp.cos(yy)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_scalar_evaluation(self):
        """Evaluation at a scalar point returns a scalar."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))
        val = f(jnp.array(0.5), jnp.array(-0.3))
        expected = jnp.cos(jnp.array(0.5) + jnp.array(-0.3))
        npt.assert_allclose(float(val), float(expected), rtol=RTOL_TIGHT)

    def test_custom_domain_evaluation(self):
        """Evaluation on a non-default domain matches direct computation."""
        domain = (0.0, 2.0, -1.0, 1.0)
        f = SeparableApprox.from_function(
            lambda x, y: jnp.sin(x) + y**2,
            domain=domain,
        )
        x_test = jnp.linspace(0.0, 2.0, 10, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        xx = xx.ravel()
        yy = yy.ravel()
        got = f(xx, yy)
        expected = jnp.sin(xx) + yy**2
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)


# ===========================================================================
# Class TestJAXContract — JIT, vmap, grad
# ===========================================================================


class TestJAXContract:
    """Tests for JAX contract: evaluation is jit/vmap/grad safe.

    JAX contract: jit=YES, vmap=YES, grad=YES (for evaluation only).
    """

    def test_jit_evaluation(self):
        """JIT-compiled evaluation matches eager evaluation."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))

        x_val = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y_val = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)

        eager = f(x_val, y_val)
        jitted = eqx.filter_jit(f)(x_val, y_val)

        npt.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-15)

    def test_vmap_evaluation(self):
        """vmap over batch of (x, y) pairs."""
        f = SeparableApprox.from_function(lambda x, y: jnp.exp(x * y))

        xs = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
        ys = jnp.linspace(-0.5, 0.5, 8, dtype=jnp.float64)

        # Use vmap to evaluate at each pair (xs[i], ys[i])
        vmapped = jax.vmap(lambda xi, yi: f(xi, yi))(xs, ys)
        expected = jnp.array([f(xs[i], ys[i]) for i in range(8)])

        npt.assert_allclose(np.array(vmapped), np.array(expected), rtol=1e-14)

    def test_grad_x(self):
        """Gradient with respect to x is available via jax.grad."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))

        # df/dx of cos(x+y) = -sin(x+y)
        x0 = jnp.array(0.3, dtype=jnp.float64)
        y0 = jnp.array(0.7, dtype=jnp.float64)

        grad_fn = jax.grad(lambda xi: f(xi, y0))
        df_dx = grad_fn(x0)
        expected = -jnp.sin(x0 + y0)
        npt.assert_allclose(float(df_dx), float(expected), rtol=1e-9)

    def test_grad_y(self):
        """Gradient with respect to y is available via jax.grad."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))

        # df/dy of cos(x+y) = -sin(x+y)
        x0 = jnp.array(0.3, dtype=jnp.float64)
        y0 = jnp.array(0.7, dtype=jnp.float64)

        grad_fn = jax.grad(lambda yi: f(x0, yi))
        df_dy = grad_fn(y0)
        expected = -jnp.sin(x0 + y0)
        npt.assert_allclose(float(df_dy), float(expected), rtol=1e-9)

    def test_jit_with_functools_partial(self):
        """JIT works when x is fixed via partial application."""
        f = SeparableApprox.from_function(lambda x, y: jnp.sin(x) * jnp.cos(y))
        x0 = jnp.array(0.5, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(f, x0))
        y_vals = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64)
        got = jitted(y_vals)
        expected = jnp.sin(x0) * jnp.cos(y_vals)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=1e-12)


# ===========================================================================
# Class TestEdgeCases — zero function, single-variable functions
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and degenerate inputs."""

    def test_zero_function(self):
        """The zero function should be representable."""
        f = SeparableApprox.from_function(lambda x, y: jnp.zeros_like(x))
        x_test = jnp.array([0.0, 0.5, -0.5], dtype=jnp.float64)
        y_test = jnp.array([0.0, 0.5, -0.5], dtype=jnp.float64)
        got = f(x_test, y_test)
        npt.assert_allclose(np.array(got), np.zeros(3), atol=1e-14)

    def test_x_only_function(self):
        """A function of x only should converge."""
        f = SeparableApprox.from_function(lambda x, y: jnp.sin(x) + jnp.zeros_like(y))
        x_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y_test = jnp.zeros(10, dtype=jnp.float64)
        got = f(x_test, y_test)
        expected = jnp.sin(x_test)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_y_only_function(self):
        """A function of y only should converge."""
        f = SeparableApprox.from_function(lambda x, y: jnp.cos(y) + jnp.zeros_like(x))
        x_test = jnp.zeros(10, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        got = f(x_test, y_test)
        expected = jnp.cos(y_test)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_high_degree_polynomial(self):
        """A 2D polynomial of moderate degree should converge."""
        def poly(x, y):
            return x**4 * y**3 - 2.0 * x**2 * y + 1.0

        f = SeparableApprox.from_function(poly)
        x_test = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
        got = f(x_test, y_test)
        expected = poly(x_test, y_test)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL, atol=1e-13)
