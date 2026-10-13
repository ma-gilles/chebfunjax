"""Scalar periodic source Newton history and preference controls.

Provenance
----------
MATLAB source: @chebop/solvebvp.m, @chebop/solvebvpNonlinear.m,
    @chebop/dampingErrorBased.m, @linop/linsolve.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The constant quadratic control has an independently captured native six-step
history; maxIter1 returns after two accepted updates. Supplemental analytic
solution bounds use1e-10, as existing scalar Newton preference controls do;
original periodic source predicates remain in test_periodic_nonlin_matlab.py.
"""
import pytest

from chebfunjax import chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop
from chebfunjax.tech.trigtech import Trigtech


@pytest.fixture(autouse=True)
def restore_session():
    previous = ChebopPref._defaults
    ChebopPref.setDefaults("factory")
    try:
        yield
    finally:
        ChebopPref._defaults = previous


def quadratic(initial=2):
    op = Chebop(lambda u: u.diff(2)+u**2, (-1., 1.))
    op.bc = "periodic"
    op.init = chebfun(initial)
    return op


def test_native_constant_history_and_representation(capsys):
    ChebopPref.setDefaults("display", "iter")
    result, info = quadratic().solvebvp(1., discretization="trigcolloc")
    assert isinstance(result.funs[0].tech, Trigtech)
    assert tuple(result.domain.breakpoints) == (-1., 1.)
    assert float((result-1).norm()) < 1e-10
    assert info["converged"]
    assert len(info["normDelta"]) == len(info["iterationHistory"]) == 6
    assert all(row["lambda"] == 1 for row in info["iterationHistory"])
    assert info["newtonTolerance"] == 200*info["bvpTol"]
    assert "Newton's method converged in 6 iterations." in capsys.readouterr().out


@pytest.mark.parametrize("session,override", [(0,None),(1,None),(1,0),(0,1)])
def test_native_maxiter_plus_one_and_override(session, override):
    ChebopPref.setDefaults("maxIter", session)
    options = {} if override is None else {"max_iter": override}
    with pytest.warns(RuntimeWarning, match="maximum number of iterations"):
        result, info = quadratic().solvebvp(1.,discretization="trigcolloc",**options)
    limit = session if override is None else override
    assert len(info["normDelta"]) == limit+1
    assert not info["converged"]
    expected = 1.25 if limit == 0 else 1.025
    assert float((result-expected).norm()) < 1e-10


def test_exact_initial_periodic_history():
    result, info = quadratic(1).solvebvp(1.,discretization="trigcolloc")
    assert info["exactInitial"]
    assert len(info["normDelta"]) == 1
    assert float((result-1).norm()) < 1e-10


def test_undamped_initial_iteration_preserves_native_no_update():
    ChebopPref.setDefaults("damping",False,"maxIter",0)
    with pytest.warns(RuntimeWarning,match="maximum number of iterations"):
        result,info = quadratic().solvebvp(1.,discretization="trigcolloc")
    assert len(info["normDelta"]) == 1
    assert info["dampingHistory"][0]["damping"] is False
    assert float((result-2).norm()) < 1e-10


def test_supported_callback_failure_is_not_silently_rerouted():
    def broken(u):
        raise ValueError("scientific callback failure")

    op = Chebop(broken)
    op.bc = "periodic"
    with pytest.raises(ValueError, match="scientific callback failure"):
        op.solve(1., discretization="trigcolloc")


def test_actual_source_first_order_selects_new_recurrence(monkeypatch):
    import math

    import jax.numpy as jnp

    from chebfunjax.operators import scalar_newton

    op = Chebop(lambda u: u.diff()-u*u.cos(), (-math.pi, math.pi))
    op.bc = "periodic"
    op.init = chebfun(lambda x: jnp.cos(x), domain=op.domain)
    captured = []

    def source_solver(*args, **kwargs):
        captured.append(kwargs)
        return 42

    monkeypatch.setattr(scalar_newton, "solve_scalar", source_solver)
    assert op.solve(1.) == 42
    assert captured[0]["backend"] == "trigcolloc"
    assert captured[0]["_periodic_data"] is not None
