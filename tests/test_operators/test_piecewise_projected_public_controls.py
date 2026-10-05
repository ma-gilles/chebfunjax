"""Independent public controls for the projected scalar BVP dispatch.

Provenance
----------
MATLAB source : @chebcolloc/reduce.m, @valsDiscretization/rhs.m,
    @chebcolloc2/toFunctionOut.m
Chebfun commit: 7574c77
Analytic polynomial oracles and predeclared bounds are independent controls,
not copied MATLAB assertion bounds.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import jump
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize("amplitude", [1.0, 1.0 + 2.0j])
@pytest.mark.parametrize("domain", [(0.0, 0.3, 1.0), (0.0, 0.2, 0.6, 1.0)])
def test_variable_coefficient_affine_quadratic_public_solution(amplitude, domain):
    # u=A*x*(1-x), with a nonzero operator offset sampled at equation nodes.
    def operator(x, u):
        return (1 + x) * u.diff(2) - x * u.diff() + 0.25 * u + (1 + x**2)

    def rhs(x):
        return amplitude * (-2.0 - 2.75 * x + 1.75 * x**2) + 1.0 + x**2

    problem = Chebop(operator, domain=domain)
    problem.lbc = problem.rbc = 0.0
    result = problem.solve(rhs, n=16)
    x = jnp.linspace(0.0, 1.0, 37)
    expected = amplitude * x * (1.0 - x)
    assert np.max(np.abs(np.asarray(result(x) - expected))) < 1000 * np.finfo(float).eps * max(1.0, abs(amplitude))
    # Direct physical residual; do not substitute a collocation matrix residual.
    residual = (1.0 + x) * result.diff(2)(x) - x * result.diff()(x) + 0.25 * result(x) + 1.0 + x**2 - rhs(x)
    assert np.max(np.abs(np.asarray(residual))) < 2e-10 * max(1.0, abs(amplitude))
    for bp in domain[1:-1]:
        assert abs(complex(jump(result, bp))) < 1e-11 * max(1.0, abs(amplitude))
        assert abs(complex(jump(result.diff(), bp))) < 1e-10 * max(1.0, abs(amplitude))


@pytest.mark.parametrize("amplitude", [1.0, 1.0 + 2.0j])
def test_cubic_public_solution_on_unequal_panels(amplitude):
    domain = (-1.0, -0.3, 1.0)
    problem = Chebop(lambda x, u: u.diff(2), domain=domain)
    problem.lbc = problem.rbc = 0.0
    result = problem.solve(lambda x: 6.0 * amplitude * x, n=16)
    x = jnp.linspace(-1.0, 1.0, 43)
    expected = amplitude * (x**3 - x)
    assert np.max(np.abs(np.asarray(result(x) - expected))) < 1000 * np.finfo(float).eps * max(1.0, abs(amplitude))
    assert np.max(np.abs(np.asarray(result.diff(2)(x) - 6.0 * amplitude * x))) < 2e-10 * max(1.0, abs(amplitude))


def test_adaptive_variable_coefficients_with_nonzero_endpoints():
    # Independent analytic control: u=2+sin(80x). No fixed n is supplied.
    def operator(x, u):
        return (1.0 + x) * u.diff(2) + (0.25 - x) * u.diff() + 0.5 * u + 1.0 + x**2

    def rhs(x):
        return (2.0 + x**2 - (6399.5 + 6400.0 * x) * jnp.sin(80.0 * x)
                + (20.0 - 80.0 * x) * jnp.cos(80.0 * x))

    domain = (0.0, 0.3, 1.0)
    problem = Chebop(operator, domain=domain)
    problem.lbc = 2.0
    problem.rbc = float(2.0 + jnp.sin(80.0))
    result = problem.solve(rhs)
    # Offset uniform probes independent of either panel's collocation grids.
    x = (jnp.arange(43) + 0.371) / 43.0
    expected = 2.0 + jnp.sin(80.0 * x)
    assert np.max(np.abs(np.asarray(result(x) - expected))) <= 3e-9
    assert abs(float(result(jnp.asarray(0.0))) - 2.0) <= 3e-9
    assert abs(float(result(jnp.asarray(1.0))) - problem.rbc) <= 3e-9
    assert abs(float(jump(result, 0.3))) <= 3e-9
    assert abs(float(jump(result.diff(), 0.3))) <= 10.0 * 5e-13 * 80.0
    residual = ((1.0 + x) * result.diff(2)(x)
                + (0.25 - x) * result.diff()(x) + 0.5 * result(x)
                + 1.0 + x**2 - rhs(x))
    # Predeclared relative backward-error bound from exact operator terms.
    scale = ((1.0 + x) * 6400.0 * jnp.abs(jnp.sin(80.0 * x))
             + jnp.abs(0.25 - x) * 80.0 * jnp.abs(jnp.cos(80.0 * x))
             + 0.5 * jnp.abs(expected) + 1.0 + x**2 + jnp.abs(rhs(x)))
    assert np.max(np.abs(np.asarray(residual))) <= 10.0 * 5e-13 * float(jnp.max(scale))
