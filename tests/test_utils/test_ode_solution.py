"""Source ODE fit tolerance and independent decaying-trajectory checks.

Provenance
----------
MATLAB source : @chebfun/odesol.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.operators.chebop import Chebop
from chebfunjax.utils.ode_solution import _ode_fit_parameters


@pytest.mark.parametrize("absolute", [1e-7, [1e-7, 0.25, 9e-7]])
def test_tolerance_follows_source_nonzero_scale_conversion(absolute):
    states = np.array([[0.5, -1.0], [0.0, 0.0], [3.0, -2.0]])
    tol, scales = _ode_fit_parameters(states, 2e-8, absolute)
    expected_scales = np.array([1.0, 0.0, 3.0])
    expected_abs = np.broadcast_to(absolute, (3,))[[0, 2]]
    expected = max(2e-8, max(expected_abs/expected_scales[[0, 2]]))
    npt.assert_array_equal(scales, expected_scales)
    npt.assert_allclose(tol, expected, rtol=2*np.finfo(float).eps, atol=0)


def test_all_zero_states_use_source_machine_epsilon_under_jit():
    tol, scales = jax.jit(_ode_fit_parameters)(jnp.zeros((2, 8)), 1e-3, 1e-2)
    assert float(tol) == np.finfo(float).eps
    npt.assert_array_equal(scales, [0.0, 0.0])


def test_complex_state_scales_use_magnitudes_without_discarding_imaginary_parts():
    states = jnp.array([[3.0+4.0j, -1.0j], [0.0+0.0j, 0.0+2.0j]])
    tol, scales = _ode_fit_parameters(states, 1e-8, jnp.array([1e-6, 8e-6]))
    npt.assert_array_equal(scales, [5.0, 2.0])
    assert float(tol) == 4e-6


@pytest.mark.parametrize("system", [False, True])
def test_global_ode_scale_resolves_decaying_tail_without_long_unhappy_funs(system):
    if system:
        problem = Chebop(lambda t, u, v: [u.diff()+u, v.diff()+2*v],
                         domain=(0.0, 30.0))
        problem.lbc = [1.0, 1.0]
        solutions = list(problem._solve_ivp_system())
        rates = [1.0, 2.0]
    else:
        problem = Chebop(lambda t, u: u.diff()+u, domain=(0.0, 30.0))
        problem.lbc = 1.0
        solutions = [problem.solve_ivp()]
        rates = [1.0]
    points = np.array([0.0, 0.37, 2.5, 10.0, 20.0, 29.5, 30.0])
    for solution, rate in zip(solutions, rates):
        npt.assert_allclose(solution(jnp.asarray(points)), np.exp(-rate*points),
                            atol=5e-10, rtol=0)
        assert all(fun.ishappy for fun in solution.funs)
        # The source scales fitting by global vscale and solver tolerance;
        # a tiny noisy tail must not produce an unresolved65537-point fit.
        assert max(len(fun) for fun in solution.funs) < 512
