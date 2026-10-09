"""Source RHS parsing controls and original initialConditions slots9/10.

Chebfun7574c77 @chebop/solveivp.m82-89, @treeVar/toFirstOrder.m79-89/223,
@chebmatrix/chebmatrix.m parseData, tests/chebop/test_initialConditions.m.
"""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref, ChebopPref
from chebfunjax.operators._coupled_ivp import UnsupportedStructure, extract
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop


@pytest.fixture(autouse=True)
def factory_preferences(monkeypatch):
    monkeypatch.setattr(ChebfunPref, '_defaults', None)
    monkeypatch.setattr(ChebopPref, '_defaults', None)
    yield


def _operator():
    n = Chebop(lambda t, u, v: [u.diff(), v.diff()], domain=(0, 1))
    n.lbc = [0, 0]
    return n


@pytest.mark.parametrize(('data', 'expected'), [
    (2., [2., 2.]), ([[2.]], [2., 2.]),
    ([1., 2.], [1., 2.]), ([[1., 2.]], [1., 2.]),
    ([[1.], [2.]], [1., 2.]),
    ([[1., 3.], [2., 4.]], [1., 2.]),
    ([1., 2., 3.], [1., 2.]),
])
def test_numeric_array_source_order_and_broadcast(data, expected):
    p = extract(_operator(), jnp.asarray(data))
    actual = jax.jit(p.rhs)(jnp.asarray(.5), jnp.zeros(2))
    assert bool(jnp.array_equal(actual, jnp.asarray(expected)))


@pytest.mark.parametrize('matrix', [False, True])
def test_nonnumeric_singleton_is_not_broadcast(matrix):
    f = cj.chebfun(2., domain=(0, 1))
    rhs = ChebMatrix([[f]], domain=(0, 1)) if matrix else f
    with pytest.raises(IndexError):
        extract(_operator(), rhs)


@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('rhs_domain', [(-1., 1.), (0., 1.+2.**-52)])
def test_public_exact_rhs_domain_error_propagates(matrix, rhs_domain):
    f = cj.chebfun(lambda t: jnp.stack([t, 2*t], axis=-1), domain=rhs_domain)
    rhs = (ChebMatrix([[f.extract_columns(0)], [f.extract_columns(1)]], domain=rhs_domain)
           if matrix else f)
    n = _operator()
    with pytest.raises(ValueError, match='CHEBFUN:CHEBOP:solveivp:domainMismatch'):
        n.solve(rhs)


def test_array_chebfun_rhs_columns_and_breakpoints():
    rhs = cj.chebfun(lambda t: jnp.stack([t, 2*t], axis=-1), domain=(0, .25, 1))
    p = extract(_operator(), rhs)
    assert p.span == (0., .25, 1.)
    with pytest.raises(UnsupportedStructure, match='row Chebfun'):
        extract(_operator(), rhs.T)
    assert bool(jnp.all(jnp.abs(p.rhs(.5, jnp.zeros(2))-jnp.asarray([.5, 1.])) <= 16*jnp.finfo(jnp.float64).eps))


def test_chebmatrix_linear_index_and_surplus_domain():
    # Column-major block order1,2,3,4; native uses first2 but merges RHS domain.
    rhs = ChebMatrix([[1., 3.], [2., 4.]], domain=(0, .25, 1))
    p = extract(_operator(), rhs)
    assert p.span == (0., .25, 1.)
    assert bool(jnp.array_equal(p.rhs(.5, jnp.zeros(2)), jnp.asarray([1., 2.])))


def test_insufficient_numeric_vector_errors():
    # Empty is not native numeric length1, so no replication.
    with pytest.raises(IndexError):
        extract(_operator(), jnp.asarray([]))


@pytest.mark.parametrize('backward', [False, True])
def test_native_initial_conditions_9_10(backward):
    n = Chebop(lambda x, u, v, w: [u.diff()+v, v.diff()+w, w.diff()+u], domain=(0, 1))
    if backward:
        n.rbc = lambda u, v, w: [u-.5, v-1.5, w+2.5]
    else:
        n.lbc = lambda u, v, w: [u-.5, v-1.5, w+2.5]
    # Native numeric column[0;1;2], no Python-list bypass of parser.
    u, v, w = n.solve(jnp.asarray([[0.], [1.], [2.]]))
    endpoint = 1. if backward else 0.
    assert abs(u(endpoint)-.5)+abs(v(endpoint)-1.5)+abs(w(endpoint)+2.5) < 5e-9
    assert n._ivp_backend_used == 'native_ode113'


def test_public_constant_vector_forcing_analytic():
    u, v = _operator().solve(jnp.asarray([[1.], [2.]]))
    x = cj.chebfun(lambda t: t, domain=(0, 1))
    # Same degree<=2 analytic100eps bound qualified by preceding polynomial gates.
    assert float((u-x).norm('inf')) < 100*jnp.finfo(jnp.float64).eps
    assert float((v-2*x).norm('inf')) < 100*jnp.finfo(jnp.float64).eps
