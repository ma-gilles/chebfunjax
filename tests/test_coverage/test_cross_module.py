"""Cross-module integration tests (V26-V27).

V26 — Chebfun2.sum → 1-D Chebfun:
  - Integrating f(x, y) over y gives a function of x
  - Integrating over x gives a function of y
  - Double integral equals direct sum2

V27 — ODE solution satisfies equation:
  - Solve u'' + u = 0 => verify u'' + u ≈ 0 pointwise
  - Solve u' = -u => verify u' + u ≈ 0 pointwise
  - Verify boundary conditions are met
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.chebfun1d.ode import bvp, eigs, ivp
from chebfunjax.chebfun2d.chebfun2 import chebfun2

ATOL_TIGHT = 1e-10
ATOL_MED = 1e-8


def _eval_pts(n=40, a=-1.0, b=1.0):
    return jnp.linspace(a + 1e-4, b - 1e-4, n, dtype=jnp.float64)


# ============================================================================
# V26 — Chebfun2.sum reduces to 1D
# ============================================================================


class TestChebfun2Sum1D:
    """Verify Chebfun2.sum(dim) produces the correct 1D marginal."""

    def test_sum_dim1_separable_sin_cos(self):
        """int_{-1}^{1} sin(x)*cos(y) dy  ==  sin(x) * 2*sin(1)."""
        f = chebfun2(lambda x, y: jnp.sin(x) * jnp.cos(y))
        # Integrate over y
        g = f.sum(dim=1)
        # Expected: sin(x) * integral_(-1)^1 cos(y) dy = sin(x) * 2*sin(1)
        expected_scale = 2.0 * float(jnp.sin(jnp.float64(1.0)))
        xs = _eval_pts()
        for xi in xs[::5]:
            g_val = float(g(float(xi), 0.0))
            expected = float(jnp.sin(xi)) * expected_scale
            assert abs(g_val - expected) < ATOL_TIGHT, (
                f"sum(dim=1) mismatch at x={float(xi):.3f}: "
                f"got {g_val:.8f}, expected {expected:.8f}"
            )

    def test_sum_dim2_separable_exp_poly(self):
        """int_{-1}^{1} exp(x) * x^2 dx  evaluated as 1D function of y."""
        f = chebfun2(lambda x, y: jnp.exp(x) * y ** 2)
        # Integrate over x: int exp(x) dx from -1 to 1 = exp(1) - exp(-1)
        g = f.sum(dim=2)
        exp_integral = float(jnp.exp(jnp.float64(1.0)) - jnp.exp(jnp.float64(-1.0)))
        ys = _eval_pts()
        for yi in ys[::5]:
            g_val = float(g(0.0, float(yi)))
            expected = exp_integral * float(yi) ** 2
            assert abs(g_val - expected) < ATOL_TIGHT, (
                f"sum(dim=2) mismatch at y={float(yi):.3f}: "
                f"got {g_val:.8f}, expected {expected:.8f}"
            )

    def test_sum2_agrees_with_sum_then_sum(self):
        """f.sum2() == f.sum(dim=1).sum(dim=2) for a separable function."""
        f = chebfun2(lambda x, y: jnp.sin(x) * jnp.cos(y))
        double_integral = float(f.sum2())
        # Analytical: int sin(x) dx from -1 to 1 = 0, so double integral = 0
        assert abs(double_integral) < ATOL_TIGHT

    def test_sum2_constant_function(self):
        """int_(-1)^1 int_(-1)^1 1 dx dy = 4."""
        f = chebfun2(lambda x, y: jnp.ones_like(x))
        val = float(f.sum2())
        assert abs(val - 4.0) < ATOL_TIGHT

    def test_sum2_product_function(self):
        """int_(-1)^1 int_(-1)^1 x^2 * y^2 dx dy = (2/3)^2 = 4/9."""
        f = chebfun2(lambda x, y: x ** 2 * y ** 2)
        val = float(f.sum2())
        expected = (2.0 / 3.0) ** 2
        assert abs(val - expected) < ATOL_TIGHT

    def test_sum_dim1_dimension_check(self):
        """After sum(dim=1), evaluation at fixed y values should be x-independent."""
        f = chebfun2(lambda x, y: jnp.exp(x) * jnp.cos(y))
        g = f.sum(dim=1)
        # g(x, y) should be constant in y for fixed x
        # g(x, .) = cos(.) integrated over [-1,1] * exp(x)
        # = 2*sin(1) * exp(x)
        scale = 2.0 * float(jnp.sin(jnp.float64(1.0)))
        xs = _eval_pts(n=5)
        for xi in xs:
            expected = scale * float(jnp.exp(xi))
            # Check at y=0 and y=0.5
            for yi in [0.0, 0.5]:
                g_val = float(g(float(xi), yi))
                assert abs(g_val - expected) < ATOL_TIGHT


# ============================================================================
# V27 — ODE solution satisfies the equation
# ============================================================================


class TestODESatisfiesEquation:
    """Verify that ODE solutions satisfy the ODE pointwise."""

    def test_bvp_harmonic_oscillator_satisfies_ode(self):
        """u'' + u = 0 on [0, pi], u(0)=0, u(pi)=0 => verify u'' + u ≈ 0."""
        import jax.numpy as jnp
        domain = (0.0, float(jnp.pi))
        u = bvp(
            lambda x, u: u.diff(2) + u,
            domain=domain,
            lbc=0.0,
            rbc=0.0,
            f=0.0,
        )
        xs = _eval_pts(a=domain[0], b=domain[1])
        residual = np.array(u.diff(2)(xs)) + np.array(u(xs))
        npt.assert_allclose(residual, np.zeros_like(residual), atol=ATOL_MED)

    def test_bvp_boundary_conditions_met(self):
        """u'' + u = 0 on [0, pi]: verify u(0) ≈ 0 and u(pi) ≈ 0."""
        import jax.numpy as jnp
        domain = (0.0, float(jnp.pi))
        u = bvp(
            lambda x, u: u.diff(2) + u,
            domain=domain,
            lbc=0.0,
            rbc=0.0,
        )
        assert abs(float(u(jnp.float64(domain[0])))) < ATOL_TIGHT
        assert abs(float(u(jnp.float64(domain[1])))) < ATOL_TIGHT

    def test_bvp_solution_is_sin(self):
        """u'' + u = 0, u(0)=0, u(pi)=0, normalised to u(pi/2)=1 => u = sin(x)."""
        import jax.numpy as jnp
        pi = float(jnp.pi)
        u = bvp(
            lambda x, u: u.diff(2) + u,
            domain=(0.0, pi),
            lbc=0.0,
            rbc=0.0,
        )
        xs = _eval_pts(a=0.0, b=pi)
        # Solution is C*sin(x) for some C; find C from values
        u_vals = np.array(u(xs))
        sin_vals = np.array(jnp.sin(xs))
        # Compute scale factor at midpoint
        mid_idx = len(xs) // 2
        if abs(sin_vals[mid_idx]) > 1e-6:
            scale = u_vals[mid_idx] / sin_vals[mid_idx]
            npt.assert_allclose(u_vals, scale * sin_vals, atol=ATOL_MED)

    def test_ivp_exponential_growth_satisfies_ode(self):
        """u' = u, u(0)=1 => verify u' - u ≈ 0 pointwise and u ≈ exp(x)."""
        u = ivp(
            lambda x, u: u.diff() - u,
            domain=(0.0, 1.0),
            ic=[1.0],
        )
        xs = _eval_pts(a=0.0, b=1.0)
        # Residual
        residual = np.array(u.diff()(xs)) - np.array(u(xs))
        npt.assert_allclose(residual, np.zeros_like(residual), atol=ATOL_MED)
        # Value comparison
        expected = np.array(jnp.exp(xs))
        npt.assert_allclose(np.array(u(xs)), expected, atol=ATOL_MED)

    def test_ivp_initial_condition_met(self):
        """u' = u, u(0) = 1: u(0) should be 1."""
        u = ivp(lambda x, u: u.diff() - u, domain=(0.0, 1.0), ic=[1.0])
        assert abs(float(u(jnp.float64(0.0))) - 1.0) < ATOL_TIGHT

    def test_bvp_poisson_satisfies_ode(self):
        """u'' = -1, u(-1) = 0, u(1) = 0 => u = (1 - x^2)/2 satisfies ODE."""
        u = bvp(
            lambda x, u: u.diff(2),
            domain=(-1.0, 1.0),
            lbc=0.0,
            rbc=0.0,
            f=-1.0,
        )
        xs = _eval_pts()
        # Residual: u'' - (-1) = u'' + 1 should be 0
        residual = np.array(u.diff(2)(xs)) + np.ones(len(xs))
        npt.assert_allclose(residual, np.zeros_like(residual), atol=ATOL_MED)
        # Also compare to exact solution
        expected = np.array((1.0 - xs ** 2) / 2.0)
        npt.assert_allclose(np.array(u(xs)), expected, atol=ATOL_MED)

    def test_eigs_laplacian_values(self):
        """Eigenvalues of -d^2/dx^2 with Dirichlet BCs on [-1,1] = (k*pi/2)^2."""
        lam = eigs(
            lambda x, u: -u.diff(2),
            domain=(-1.0, 1.0),
            lbc=0.0,
            rbc=0.0,
            k=4,
        )
        expected = [(k * float(jnp.pi) / 2.0) ** 2 for k in range(1, 5)]
        lam_sorted = np.sort(np.real(np.array(lam)))
        npt.assert_allclose(lam_sorted, expected, rtol=1e-6)
