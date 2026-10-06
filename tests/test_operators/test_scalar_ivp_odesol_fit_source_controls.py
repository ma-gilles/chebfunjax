"""Source controls for scalar solve_ivp -> ODESOL representation fitting.

Provenance
----------
MATLAB sources: @chebop/solveivp.m, @chebfun/constructODEsol.m,
    @chebfun/odesol.m
Chebfun commit: 7574c77
Independent solution oracle: u(t) = exp(t) for u' - u = 0.

The analytic tolerances here are focused Python diagnostic bounds for the
SciPy marcher and source-derived fit path. They are not replacements for or
claims about any original MATLAB test bound.
"""

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


def _capture_solver_and_odesol(monkeypatch):
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
    return solver_results, fit_calls


@pytest.mark.parametrize("reverse", [False, True])
def test_restarted_fit_uses_ordered_callbacks_histories_and_checker(
        monkeypatch, reverse):
    checker = "strict" if reverse else "classic"
    monkeypatch.setattr(ChebopPref, "_defaults", ChebopPref(happinessCheck=checker))
    solver_results, fit_calls = _capture_solver_and_odesol(monkeypatch)
    problem = Chebop(
        lambda t, u: u.diff() - u,
        domain=(0.0, 0.5, 1.0),
    )
    problem.ivp_method = "LSODA"  # This test observes the explicit SciPy adapter.
    problem.ivp_restart_solver = True
    if reverse:
        problem.rbc = math.e
    else:
        problem.lbc = 1.0

    # The original supplemental fixture requested rtol=8e-9, atol=1e-10:
    # raw LSODA dense error reached 3.197e-8 in reverse, before fitting.
    # Request tighter integration precision so the unchanged 2e-8 analytic
    # bound checks conversion behavior. This is not a MATLAB test change.
    rtol, atol = 1e-10, 1e-12
    solution = problem.solve_ivp(rtol=rtol, atol=atol)

    assert len(solver_results) == 2
    assert len(fit_calls) == 1
    call = fit_calls[0]
    assert isinstance(call["interpolants"], list)
    assert len(call["interpolants"]) == 2
    expected_domain = ((1.0, 0.5, 0.0) if reverse
                       else (0.0, 0.5, 1.0))
    assert call["domain"] == expected_domain
    expected_states = np.concatenate([result.y for result in solver_results],
                                     axis=1)
    npt.assert_array_equal(call["states"], expected_states)
    assert call["relative_tolerance"] == rtol
    assert call["absolute_tolerance"] == atol
    assert call["check"] == checker == ChebopPref().happinessCheck

    points = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    npt.assert_allclose(solution(jnp.asarray(points)), np.exp(points),
                        rtol=0.0, atol=2e-8)
    break_index = solution.domain.breakpoints.index(0.5)
    npt.assert_allclose(solution.point_values[break_index], math.exp(0.5),
                        rtol=0.0, atol=2e-8)


def test_single_dense_fit_records_callback_point_values_and_tolerances(
        monkeypatch):
    solver_results, fit_calls = _capture_solver_and_odesol(monkeypatch)
    problem = Chebop(lambda t, u: u.diff() - u, domain=(0.0, 1.0))
    problem.ivp_method = "LSODA"  # This test observes the explicit SciPy adapter.
    problem.ivp_restart_solver = False
    problem.lbc = 1.0

    rtol, atol = 3e-9, 7e-11
    solution = problem.solve_ivp(rtol=rtol, atol=atol)

    assert len(solver_results) == 1
    assert len(fit_calls) == 1
    call = fit_calls[0]
    assert callable(call["interpolants"])
    assert call["domain"] == (0.0, 1.0)
    npt.assert_array_equal(call["states"], solver_results[0].y)
    assert call["relative_tolerance"] == rtol
    assert call["absolute_tolerance"] == atol
    expected_endpoints = solver_results[0].sol(np.array([0.0, 1.0]))[0]
    npt.assert_array_equal(solution.point_values, expected_endpoints)
    npt.assert_allclose(solution(jnp.asarray([0.25, 0.75])),
                        np.exp([0.25, 0.75]), rtol=0.0, atol=2e-8)


def test_terminal_event_fits_clipped_history_before_nan_padding(monkeypatch):
    solver_results, fit_calls = _capture_solver_and_odesol(monkeypatch)
    problem = Chebop(lambda t, u: u.diff() - u, domain=(0.0, 1.5))
    problem.ivp_method = "LSODA"  # This test observes the explicit SciPy adapter.
    problem.ivp_restart_solver = False
    problem.lbc = 1.0
    problem.maxnorm = 2.0

    solution = problem.solve_ivp(rtol=1e-10, atol=1e-12)

    assert len(solver_results) == 1
    assert len(fit_calls) == 1
    call = fit_calls[0]
    assert callable(call["interpolants"])
    assert call["domain"][0] == 0.0
    cutoff = call["domain"][-1]
    npt.assert_allclose(cutoff, math.log(2.0), rtol=0.0, atol=1e-9)
    npt.assert_array_equal(call["states"], solver_results[0].y)
    assert cutoff < 1.5

    before = np.array([0.1, 0.3, 0.5])
    npt.assert_allclose(solution(jnp.asarray(before)), np.exp(before),
                        rtol=0.0, atol=2e-8)
    after = np.asarray(solution(jnp.asarray([1.0, 1.4])))
    assert np.isnan(after).all()
    assert np.isfinite(float(solution(jnp.asarray(0.25))))


def test_reverse_terminal_event_fits_clipped_history_before_nan_padding(monkeypatch):
    solver_results, fit_calls = _capture_solver_and_odesol(monkeypatch)
    problem = Chebop(lambda t, u: u.diff() + u, domain=(0.0, 1.5))
    problem.ivp_method = "LSODA"  # This test observes the explicit SciPy adapter.
    problem.ivp_restart_solver = False
    problem.rbc = 1.0
    problem.maxnorm = 2.0

    solution = problem.solve_ivp(rtol=1e-10, atol=1e-12)

    assert len(solver_results) == len(fit_calls) == 1
    call = fit_calls[0]
    assert callable(call["interpolants"])
    assert call["domain"][0] == 1.5
    cutoff = call["domain"][-1]
    npt.assert_allclose(cutoff, 1.5 - math.log(2.0), rtol=0.0, atol=1e-9)
    npt.assert_array_equal(call["states"], solver_results[0].y)
    points = jnp.array([1.0, 1.2, 1.4])
    npt.assert_allclose(solution(points), np.exp(1.5 - np.asarray(points)),
                        rtol=0.0, atol=2e-8)
    assert np.isnan(np.asarray(solution(jnp.array([0.1, 0.5])))).all()
    assert np.isfinite(float(solution(jnp.asarray(1.25))))
