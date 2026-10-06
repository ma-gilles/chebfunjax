"""Forward/reverse ODESOL controls for coupled IVP event fits.

Provenance
----------
MATLAB sources: @chebop/solveivp.m, @chebfun/constructODEsol.m,
    @chebfun/odesol.m
Chebfun commit: 7574c77
Independent solution oracle: u(t)=exp(±t), v(t)=exp(±2t).

The value and event-location tolerances are focused Python diagnostic
bounds, not original MATLAB assertions or claims of integrator parity.
"""

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize("reverse", [False, True])
def test_coupled_event_fit_clips_domain_then_pads_each_component(
        monkeypatch, reverse):
    import scipy.integrate

    from chebfunjax.utils import ode_solution

    solver_results = []
    fit_calls = []
    original_solver = scipy.integrate.solve_ivp
    original_fit = ode_solution._odesol_from_dense

    def record_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        solver_results.append(result)
        return result

    def record_fit(interpolants, domain, states, relative_tolerance,
                   absolute_tolerance, *, check="standard"):
        fit_calls.append({
            "interpolants": interpolants,
            "domain": tuple(domain),
            "states": np.asarray(states).copy(),
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "check": check,
        })
        return original_fit(
            interpolants, domain, states, relative_tolerance,
            absolute_tolerance, check=check,
        )

    monkeypatch.setattr(scipy.integrate, "solve_ivp", record_solver)
    monkeypatch.setattr(ode_solution, "_odesol_from_dense", record_fit)

    if reverse:
        # March from t=1 to t=-1; both variables grow along the march.
        domain = (-1.0, 1.0)
        problem = Chebop(
            lambda t, u, v: [u.diff() + u, v.diff() + 2*v],
            domain=domain,
        )
        problem.rbc = [math.exp(-1.0), math.exp(-2.0)]
        finite_points = np.array([1.0, 0.4, 0.0, -0.5])
        nan_points = np.array([-0.8, -1.0])
        expected_u = np.exp(-finite_points)
        expected_v = np.exp(-2*finite_points)
        expected_cutoff = -math.log(2.0)
    else:
        # March from t=0 to t=1.5; both variables grow along the march.
        domain = (0.0, 1.5)
        problem = Chebop(
            lambda t, u, v: [u.diff() - u, v.diff() - 2*v],
            domain=domain,
        )
        problem.lbc = [1.0, 1.0]
        finite_points = np.array([0.0, 0.1, 0.3, 0.5])
        nan_points = np.array([1.0, 1.4])
        expected_u = np.exp(finite_points)
        expected_v = np.exp(2*finite_points)
        expected_cutoff = math.log(2.0)

    # Only the first state is monitored; the second maxnorm is disabled.
    problem.maxnorm = [2.0, np.inf]
    rtol, atol = 1e-10, 1e-12
    problem.ivp_reltol = rtol
    problem.ivp_abstol = atol
    solutions = list(problem._solve_ivp_system())

    assert len(solver_results) == 1
    assert len(fit_calls) == 1
    call = fit_calls[0]
    assert callable(call["interpolants"])
    assert call["domain"][0] == (1.0 if reverse else 0.0)
    cutoff = call["domain"][-1]
    npt.assert_allclose(cutoff, expected_cutoff, rtol=0.0, atol=1e-9)
    assert call["domain"][0] != call["domain"][-1]
    npt.assert_array_equal(call["states"], solver_results[0].y)
    assert call["relative_tolerance"] == rtol
    assert call["absolute_tolerance"] == atol
    assert call["check"] == ChebopPref().happinessCheck

    assert len(solutions) == 2
    npt.assert_allclose(solutions[0](jnp.asarray(finite_points)), expected_u,
                        rtol=0.0, atol=2e-8)
    npt.assert_allclose(solutions[1](jnp.asarray(finite_points)), expected_v,
                        rtol=0.0, atol=2e-8)
    assert np.isnan(np.asarray(solutions[0](jnp.asarray(nan_points)))).all()
    assert np.isnan(np.asarray(solutions[1](jnp.asarray(nan_points)))).all()
