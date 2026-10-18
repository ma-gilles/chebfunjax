"""Bounded coupled derivative towers: native7574 toFirstOrder/sortConditions.

This qualifies unit diagonal highest derivatives with position-only nonlinear
terms, not the full treeVar grammar or native chaotic trajectory parity.
"""
import importlib

import jax
import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.operators._coupled_ivp import (
    UnsupportedStructure,
    classify_local_operator,
    extract,
    prepare,
    solve,
)
from chebfunjax.operators.chebop import Chebop

EPS = jnp.finfo(jnp.float64).eps


def _polynomial(backward=False):
    op = Chebop(lambda t, u, v: [u.diff(2)-(2+4j), v.diff()-3j],
                domain=(0., 1.))
    if backward:
        op.rbc = lambda u, v: [v-(2+3j), u.diff()-(3+3j), u-(4+4j)]
    else:
        op.lbc = lambda u, v: [v-2, u.diff()-(1-1j), u-(2+3j)]
    return op


def test_original_threeplanets_extraction_and_jitted_rhs():
    def planetfun(t, x, y, z):
        forceYX = (y-x)/abs(y-x)**3
        forceZX = (z-x)/abs(z-x)**3
        forceZY = (z-y)/abs(z-y)**3
        return [x.diff(2)-forceYX-forceZX,
                y.diff(2)+forceYX-forceZY,
                z.diff(2)+forceZX+forceZY]
    op = Chebop(planetfun, domain=(0., 100.))
    op.lbc = lambda x, y, z: [x, y-3, z-4j, x.diff(), y.diff(), z.diff()]
    assert classify_local_operator(op) is None
    plan = extract(op)
    assert plan.orders == (2, 2, 2)
    assert plan.var_index == (0, 2, 4)
    assert plan.span == (0., 100.)
    assert plan.initial.dtype == jnp.complex128
    assert jnp.array_equal(plan.initial, jnp.array([0., 0., 3., 0., 4j, 0.]))
    actual = jax.jit(plan.rhs)(jnp.asarray(0.), plan.initial)
    expected = jnp.array([0., 1/9+1j/16, 0., -1/9-3/125+4j/125,
                          0., 3/125-1j/16-4j/125])
    assert jnp.max(jnp.abs(actual-expected)) < 100*EPS


@pytest.mark.parametrize('backward', [False, True])
def test_complex_polynomial_public_native_and_full_tower(monkeypatch, backward):
    from chebfunjax.utils import ode_solution
    original = ode_solution._odesol_from_dense
    state_shapes = []
    def capture(interpolants, domain, states, *args, **kwargs):
        state_shapes.append(states.shape)
        assert states.dtype == jnp.complex128
        return original(interpolants, domain, states, *args, **kwargs)
    monkeypatch.setattr(ode_solution, '_odesol_from_dense', capture)
    op = _polynomial(backward)
    plan = extract(op)  # baseline rejects before any legacy numerical solve
    assert plan.orders == (2, 1) and plan.var_index == (0, 2)
    u, v = op.solve(0.)
    assert op._ivp_backend_used == 'native_ode113'
    assert state_shapes and all(shape[0] == 3 for shape in state_shapes)
    points = jnp.linspace(0., 1., 41)
    want_u = 2+3j+(1-1j)*points+(1+2j)*points**2
    want_v = 2+3j*points
    assert jnp.max(jnp.abs(u(points).reshape(-1)-want_u)) < 100*EPS
    assert jnp.max(jnp.abs(v(points).reshape(-1)-want_v)) < 100*EPS
    endpoint = 1. if backward else 0.
    assert jnp.abs(u(endpoint).item()-(4+4j if backward else 2+3j)) < 100*EPS
    assert jnp.abs(u.diff()(endpoint).item()-(3+3j if backward else 1-1j)) < 100*EPS


def test_tower_maxnorm_watches_positions_and_projects_after_fitting(monkeypatch):
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    op = Chebop(lambda t, u, v: [u.diff(2), v.diff()], domain=(0., 1.))
    op.lbc = lambda u, v: [u, u.diff()-10, v]
    op.maxnorm = 1.
    plan = extract(op)
    selections = []
    class Output:
        def extract_columns(self, k):
            selections.append(k)
            return chebfun(k, domain=(0., 1.))
    def fake(rhs, span, initial, options, *, backend):
        assert backend == 'native' and initial.shape == (3,)
        assert jnp.array_equal(initial, jnp.array([0., 10., 0.]))
        value, terminal, direction = options['Events'](0., jnp.array([.5, 10., 0.]))
        assert jnp.array_equal(value, jnp.array([-.5, -1.]))
        assert jnp.all(terminal == 1) and jnp.all(direction == 0)
        return Output()
    monkeypatch.setattr(module, 'ode113', fake)
    solve(op, plan)
    assert selections == [0, 2]


@pytest.mark.parametrize('kind', ['velocity', 'mixed_highest', 'nonlinear_highest', 'delay'])
def test_unsupported_tower_grammar_is_not_marched(kind):
    operators = {
        'velocity': lambda t, u, v: [u.diff(2)+u.diff(), v.diff()],
        'mixed_highest': lambda t, u, v: [u.diff(2)+v.diff(2), v.diff(2)],
        'nonlinear_highest': lambda t, u, v: [u.diff(2)**2, v.diff()],
        'delay': lambda t, u, v: [u.diff(2)+u(t/2), v.diff()],
    }
    op = Chebop(operators[kind], domain=(0., 1.))
    op.lbc = lambda u, v: [u, u.diff(), v]
    assert classify_local_operator(op) is None
    with pytest.raises(UnsupportedStructure):
        extract(op)


def test_numeric_endpoint_vectors_do_not_define_coupled_derivative_towers():
    # Native parseBC.m22-35,128-138 maps numeric system BCs to values of
    # unknowns only; it explicitly does not infer higher-order tower states.
    op = _polynomial()
    for values in ([2+3j, 1-1j, 2], [2, 2]):
        op.lbc = values
        with pytest.raises(UnsupportedStructure, match='numeric endpoint'):
            extract(op)
        assert prepare(op) is None
        with pytest.raises(UnsupportedStructure, match='numeric endpoint'):
            prepare(op, selected='ode113')
