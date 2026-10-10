"""Structural/action controls; native Lorenz predicate retained separately.

Native sources: Chebfun 7574c77 @chebop/solveivp.m and
 tests/chebop/test_LorenzIVP.m. Kernel provider is R2025b native_ode113.
No runtime MATLAB comparison is implied by shared-provider route controls.
"""
import importlib

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators._coupled_ivp import UnsupportedStructure, extract, prepare, solve
from chebfunjax.operators.chebop import Chebop

EPS = 2.220446049250313e-16


@pytest.fixture(autouse=True)
def factory_preferences(monkeypatch):
    from chebfunjax.chebpref import ChebfunPref, ChebopPref
    # monkeypatch restores each exact prior object even after test failure.
    monkeypatch.setattr(ChebfunPref, '_defaults', None)
    monkeypatch.setattr(ChebopPref, '_defaults', None)
    pref = ChebopPref()
    assert pref.ivpSolver == 'ode113'
    assert pref.ivpAbsTol == 1e5*EPS and pref.ivpRelTol == 100*EPS
    yield


def _polynomial(backward=False):
    n = Chebop(lambda t, u, v: [u.diff()-1, v.diff()-2*t], domain=(0, 1))
    if backward:
        n.rbc = [1, 1]
    else:
        n.lbc = [0, 0]
    return n


@pytest.mark.parametrize('backward', [False, True])
def test_structural_polynomial_jit(backward):
    p = extract(_polynomial(backward))
    assert p.span == ((1., 0.) if backward else (0., 1.))
    actual = jax.jit(p.rhs)(jnp.asarray(.25), jnp.asarray([3., 7.]))
    assert bool(jnp.array_equal(actual, jnp.asarray([1., .5])))


def test_reordered_boundary():
    n = Chebop(lambda t, u, v, w: [u.diff(), v.diff(), w.diff()], domain=(0, 5))
    n.lbc = lambda u, v, w: [w-20, v+15, u+14]
    assert bool(jnp.array_equal(extract(n).initial, jnp.asarray([-14., -15., 20.])))


def test_forcing_breakpoints_and_nonlinear_action():
    f = cj.chebfun(lambda t: t, domain=(0, .25, 1))
    g = cj.chebfun(lambda t: 2*t, domain=(0, .5, 1))
    n = Chebop(lambda t, u, v: [u.diff()+f+3*(u-v)*(-(u-v)**2).exp(),
                               v.diff()+g+3*(v-u)*(-(v-u)**2).exp()], domain=(0, 1))
    n.lbc = [1, -1]
    p = extract(n)
    assert p.span == (0., .25, .5, 1.)
    for t in (.125, .375, .75):
        y = jnp.asarray([.75, -.25])
        actual = jax.jit(p.rhs)(jnp.asarray(t), y)
        expected = jnp.asarray([-t-3*jnp.exp(-1.), -2*t+3*jnp.exp(-1.)])
        # At these dyadic inputs f/g are exact degree1 polynomials; allowance
        # covers transform/evaluation and elementary exp arithmetic, not ODE error.
        assert bool(jnp.all(jnp.abs(actual-expected) <= 64*EPS*jnp.maximum(1., jnp.abs(expected))))


@pytest.mark.parametrize('kind', ['nonlinear_derivative', 'mixed', 'higher', 'delay', 'branch'])
def test_structural_rejections(kind):
    ops = {
        'nonlinear_derivative': lambda t, u, v: [u.diff()**2, v.diff()],
        'mixed': lambda t, u, v: [u.diff()+v.diff(), v.diff()],
        'higher': lambda t, u, v: [u.diff(2), v.diff()],
        'delay': lambda t, u, v: [u.diff()+u(t/2), v.diff()],
        'branch': lambda t, u, v: [u.diff()+(u if bool(u) else v), v.diff()],
    }
    n = Chebop(ops[kind], domain=(0, 1))
    n.lbc = [0, 0]
    with pytest.raises(UnsupportedStructure):
        extract(n)
    assert prepare(n) is None
    with pytest.raises(UnsupportedStructure):
        prepare(n, selected='ode113')


def test_explicit_alternate_selection_preserved():
    assert prepare(_polynomial(), selected='RK45') is None


def test_events_structurally_native():
    # Updated added-Python scope predicate: events now have a native route.
    # This checks extraction only, not scalar terminal flag compatibility.
    n = _polynomial()
    n.maxnorm = 10
    assert prepare(n) is not None
    assert prepare(n, selected='ode113') is not None


def test_one_array_call_and_options(monkeypatch):
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    n = _polynomial()
    n.ivp_reltol = 1e-8
    n.ivp_abstol = 1e-10
    n.ivp_restart_solver = False
    calls = []
    class Output:
        def extract_columns(self, k):
            return cj.chebfun(k, domain=(0, 1))
    def fake(rhs, span, initial, options, *, backend):
        calls.append((span, initial, options, backend))
        return Output()
    monkeypatch.setattr(module, 'ode113', fake)
    result = solve(n, extract(n))
    assert len(calls) == 1 and len(result) == 2
    assert calls[0][2]['RelTol'] == 1e-8
    assert calls[0][2]['AbsTol'] == 1e-10
    assert calls[0][2]['restartSolver'] is False
    assert calls[0][3] == 'native'


@pytest.mark.parametrize('exception', [RuntimeError, KeyboardInterrupt])
def test_public_native_failure_propagates(monkeypatch, exception):
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    def fail(*args, **kwargs):
        raise exception('actual native boundary failure')
    monkeypatch.setattr(module, 'ode113', fail)
    n = _polynomial()
    with pytest.raises(exception, match='actual native boundary failure'):
        n.solve(0)
    assert n._ivp_backend_used == 'native_ode113'


@pytest.mark.parametrize('backward', [False, True])
def test_public_polynomial_analytic(backward):
    n = _polynomial(backward)
    u, v = n.solve(0)
    # Same polynomial independent-control bound as accepted native Adams
    # controls: degree<=2 represented exactly apart from arithmetic roundoff.
    assert float((u-cj.chebfun(lambda t: t, domain=(0, 1))).norm('inf')) < 100*EPS
    assert float((v-cj.chebfun(lambda t: t*t, domain=(0, 1))).norm('inf')) < 100*EPS
    assert n._ivp_backend_used == 'native_ode113'
