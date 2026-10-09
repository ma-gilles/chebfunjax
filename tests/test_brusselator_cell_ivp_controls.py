"""Public cell grammar, continuous container norm and preference controls.

Source context: Chebfun7574c77 @chebop/solveivp.m and @chebmatrix/norm.m.
These six controls do not qualify all native cell/forcing/preference inputs.
"""
import importlib

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref, ChebopPref
from chebfunjax.operators._coupled_ivp import extract, solve
from chebfunjax.operators.chebmatrix import ChebMatrix
from tests.test_matlab_port.chebop.test_ivp_chebmatrix_syntax_matlab import DOM, _cell


@pytest.fixture(autouse=True)
def factory_preferences(monkeypatch):
    monkeypatch.setattr(ChebfunPref, '_defaults', None)
    monkeypatch.setattr(ChebopPref, '_defaults', None)
    yield


@pytest.mark.parametrize('backward', [False, True])
@pytest.mark.parametrize('callable_bc', [False, True])
def test_actual_cell_constructor_and_endpoint_setter(backward, callable_bc):
    n = _cell()
    values = [.7, .9, 1.2] if backward else [1, 3, 4]
    if callable_bc:
        def condition(u):
            return [u[0]-values[0], u[1]-values[1], u[2]-values[2]]
    else:
        condition = values
    if backward:
        n.rbc = condition
    else:
        n.lbc = condition
    assert n._n_vars() == 3
    p = extract(n)
    assert bool(jnp.array_equal(p.initial, jnp.asarray(values)))
    assert p.span == ((5., 0.) if backward else (0., 5.))
    actual = jax.jit(p.rhs)(jnp.asarray(.25), jnp.asarray([1., 2., 3.]))
    # Integer/dyadic inputs: these short polynomial sums are exactly representable.
    assert bool(jnp.array_equal(actual, jnp.asarray([-1., 1., -1.5])))


def test_public_matrix_difference_uses_frobenius_norm():
    a = ChebMatrix([[cj.chebfun(3., domain=DOM)],
                    [cj.chebfun(4., domain=DOM)]], domain=DOM)
    b = ChebMatrix([[cj.chebfun(0., domain=DOM)],
                    [cj.chebfun(0., domain=DOM)]], domain=DOM)
    expected = 5*jnp.sqrt(5.)  # sqrt(integral_0^5 (3²+4²)).
    assert abs((a-b).norm()-float(expected)) <= 16*jnp.finfo(jnp.float64).eps*expected
    assert (a-a).norm() == 0


def test_session_preferences_and_instance_override(monkeypatch):
    monkeypatch.setattr(ChebopPref, '_defaults', ChebopPref({
        'ivpRelTol': 1e-8, 'ivpAbsTol': 2e-10,
        'ivpRestartSolver': False, 'happinessCheck': 'strict'}))
    n = _cell()
    n.lbc = [1, 3, 4]
    n.ivp_abstol = 3e-10
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    captured = []
    class Output:
        def extract_columns(self, k):
            return cj.chebfun(k, domain=DOM)
    def observe(rhs, span, initial, options, *, backend):
        captured.append(options)
        return Output()
    monkeypatch.setattr(module, 'ode113', observe)
    solve(n, extract(n))
    assert captured == [{'RelTol': 1e-8, 'AbsTol': 3e-10,
                         'restartSolver': False, 'happinessCheck': 'strict'}]
