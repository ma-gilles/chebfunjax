"""Public output selection; native @chebop/linearize.m, pin7574c77.

Native L is always the first output; established Python explicit-state calls
return that operator, while omitted-state calls retain the information triple.
These polynomial action controls supplement the unchanged native test.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('state_kind', ['list', 'tuple', 'chebmatrix'])
def test_explicit_coupled_state_returns_operator_with_correct_action(state_kind):
    domain = (-1., .2, 1.)
    u = chebfun(lambda x: 1+x, domain=domain)
    v = chebfun(lambda x: x*x, domain=domain)
    h = chebfun(lambda x: x*x-1, domain=domain)
    k = chebfun(lambda x: 2*x+3, domain=domain)
    state = {'list': [u, v], 'tuple': (u, v),
             'chebmatrix': ChebMatrix([[u], [v]])}[state_kind]
    op = Chebop(lambda x, u, v: [u.diff()+u*v, v.diff(2)+u*u], domain=domain)
    operator = op.linearize(state)
    assert isinstance(operator, BlockLinop)
    result = operator * [h, k]
    expected = [h.diff()+h*v+u*k, k.diff(2)+2*u*h]
    for actual, wanted in zip(result, expected, strict=True):
        assert (actual-wanted).norm(jnp.inf) < 2e-12


def test_explicit_zero_state_is_operator_and_ignores_init():
    op = Chebop(lambda x, u, v: [u.diff()+u*v, v.diff()+u*u])
    op.init = [3., 4.]
    operator = op.linearize([0., 0.])
    assert isinstance(operator, BlockLinop)
    h = chebfun(lambda x: x*x)
    k = chebfun(lambda x: 1+3*x)
    result = operator * [h, k]
    assert (result[0]-h.diff()).norm(jnp.inf) < 2e-12
    assert (result[1]-k.diff()).norm(jnp.inf) < 2e-12
