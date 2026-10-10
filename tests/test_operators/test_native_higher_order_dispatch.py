"""Higher-order native IVP routing: source state order and analytic controls."""

import importlib
import math

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize("order,reverse", [(2, False), (2, True), (3, False)])
def test_native_derivative_tower_and_orientation(monkeypatch, order, reverse):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    original = module.ode113
    calls = []

    def observed(fun, span, initial, options, **kwargs):
        calls.append((fun, span, initial, options, kwargs))
        return original(fun, span, initial, options, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("higher-order native IVP attempted fallback")

    monkeypatch.setattr(module, "ode113", observed)
    monkeypatch.setattr(scipy.integrate, "solve_ivp", forbidden)
    for method in (
        "_solve_linear",
        "_solve_nonlinear",
        "_solve_ivp_system_highorder",
        "_solve_piecewise",
    ):
        monkeypatch.setattr(Chebop, method, forbidden)
    if order == 2:
        problem = Chebop(lambda t, u: u.diff(2) + u, domain=(0.0, 1.0))
        initial = [math.cos(1), -math.sin(1)] if reverse else [1.0, 0.0]

        def expected(x):
            return jnp.cos(x)
    else:
        problem = Chebop(lambda t, u: u.diff(3) - u, domain=(0.0, 1.0))
        initial = [1.0, 1.0, 1.0]

        def expected(x):
            return jnp.exp(x)

    if reverse:
        problem.rbc = initial
    else:
        problem.lbc = initial
    result = problem.solve(0.0)
    assert problem._ivp_backend_used == "native_ode113" and len(calls) == 1
    fun, span, y0, options, kwargs = calls[0]
    assert tuple(span) == ((1.0, 0.0) if reverse else (0.0, 1.0))
    np.testing.assert_array_equal(y0, initial)
    assert y0.shape == (order,) and kwargs == {"backend": "native"}
    probe = jnp.arange(1.0, order + 1.0)
    np.testing.assert_array_equal(
        fun(jnp.asarray(0.25), probe),
        jnp.asarray([2.0, -1.0]) if order == 2 else jnp.asarray([2.0, 3.0, 1.0]),
    )
    assert options["RelTol"] == 100 * np.finfo(float).eps
    assert options["AbsTol"] == 1e5 * np.finfo(float).eps
    assert options["restartSolver"] is True
    x = jnp.asarray([0.0, 0.125, 0.5, 0.875, 1.0])
    # Added analytic bound inherited from existing scalar dispatch controls;
    # not a replacement for any original MATLAB test predicate.
    np.testing.assert_allclose(result(x), expected(x), rtol=0, atol=2e-8)


def test_native_higher_order_failure_propagates(monkeypatch):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")

    def fail(*args, **kwargs):
        raise RuntimeError("native sentinel")

    monkeypatch.setattr(module, "ode113", fail)
    problem = Chebop(lambda t, u: u.diff(2) + u, domain=(0.0, 1.0))
    problem.lbc = [1.0, 0.0]
    with pytest.raises(RuntimeError, match="native sentinel"):
        problem.solve(0.0)
    assert problem._ivp_backend_used == "native_ode113"
