"""Independent public BVP regression for Breakpoints Problem A.

This is an independently derived regression, not source MATLAB test output. The analytic formula is derived from the ODE and zero Dirichlet data.

Provenance
----------
MATLAB example: ``examples/ode-linear/Breakpoints.m``, Problem A, commit
``f4b9ea46cfc2f52f20a844627f4a74d0bb10098c`` (example checkout). Source has no pointwise
assertion tolerance; ``VALUE_LIMIT`` below is a proposed independent
regression bound, not a copied MATLAB assertion.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import jump
from chebfunjax.operators.chebop import Chebop

EPSILON = 1.0e-8
BREAK = 40.0 * EPSILON
DOMAIN = (0.0, BREAK, 1.0)
# Frozen independently selected interior probes from the projected diagnostic.
PROBES = np.asarray((
    1.25e-9,
    1.375e-8,
    7.250000000000001e-8,
    2.2125e-7,
    3.9875e-7,
    0.0001373999452,
    0.013700394519999999,
    0.1370003452,
    0.5310001876,
    0.9130000348,
))
# This independent value bound was frozen before the baseline and remains
# unchanged. It is not an assertion from the MATLAB example.
VALUE_LIMIT = 1.0e-9
# Source truncation controls function accuracy, not absolute high derivatives.
# A 90-digit analytic coefficient oracle demonstrates that even exact
# 39-term output has residual1.36e-5 and derivative error1.40e-5. Use
# global operator scales for these independent backward-error checks.
BVP_TOL = 5e-13
RESIDUAL_LIMIT = 10.0 * BVP_TOL * (2.0 / EPSILON + 1.0)
TRACE_LIMIT = 1.0e-9
DERIVATIVE_TRACE_LIMIT = 10.0 * BVP_TOL / EPSILON


def _analytic(x):
    """Exact solution of -eps*u''-u'=1, u(0)=u(1)=0."""
    x = jnp.asarray(x)
    boundary_layer = -jnp.expm1(-x / EPSILON)
    endpoint_normalizer = -jnp.expm1(-1.0 / EPSILON)
    return boundary_layer / endpoint_normalizer - x


def test_public_piecewise_breakpoints_boundary_layer_against_exact_solution():
    """Public dispatch must retain the n+2/raw-to-projected source layout."""
    problem = Chebop(
        lambda x, u: -EPSILON * u.diff(2) - u.diff(),
        domain=DOMAIN,
    )
    problem.lbc = 0.0
    problem.rbc = 0.0
    solution = problem.solve(1.0)

    got = np.asarray(solution(jnp.asarray(PROBES)))
    expected = np.asarray(_analytic(jnp.asarray(PROBES)))
    assert np.max(np.abs(got - expected)) <= VALUE_LIMIT

    assert abs(float(solution(jnp.asarray(DOMAIN[0])))) <= TRACE_LIMIT
    assert abs(float(solution(jnp.asarray(DOMAIN[-1])))) <= TRACE_LIMIT
    assert abs(float(jump(solution, BREAK))) <= TRACE_LIMIT
    assert abs(float(jump(solution.diff(), BREAK))) <= DERIVATIVE_TRACE_LIMIT

    residual = (-EPSILON * solution.diff(2) - solution.diff() - 1.0)
    residual_values = np.asarray(residual(jnp.asarray(PROBES)))
    assert np.max(np.abs(residual_values)) <= RESIDUAL_LIMIT
