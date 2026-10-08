"""Five original MATLAB seed assertions and independent-variable controls.

Provenance: tests/adchebfun/test_seed.m, @adchebfun/adchebfun.m (seed),
@chebmatrix/mtimes.m; Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Original domain [-2,4] and tolerance1e-15 remain unchanged. No RNG involved.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc, OperatorBlock
from chebfunjax.operators.chebmatrix import ChebMatrix

DOM = (-2., 4.)
TOL = 1e-15


def _x():
    return chebfun(lambda x: x, domain=DOM)


def test_source_seed_identity_operator():
    x = _x()
    result = ADChebfun(x).seed(1, 1)
    assert isinstance(result.jacobian, OperatorBlock)
    assert float((result.jacobian.apply(x)-x).norm()) < TOL
    assert result.linearity == (True,) and result.is_linear


def test_source_seed_identity_function():
    result = ADChebfun(_x()).seed(1, 0)
    assert isinstance(result.jacobian, Chebfun)
    assert float((result.jacobian-1).norm()) < TOL


def test_source_seed_shortcut_three_functions():
    x = _x()
    result = ADChebfun(x).seed(2, 3)
    row = result.jacobian.blocks[0]
    error = ((row[1].apply(x)-x).norm()+row[0].apply(x).norm()+row[2].apply(x).norm())
    assert float(error) < TOL
    assert result.linearity == (True, True, True)


def test_source_seed_mixed_scalar_identity():
    x = _x()
    result = ADChebfun(x).seed(3, [True, False, False, True])
    row = result.jacobian.blocks[0]
    error = row[0].apply(x).norm()+row[1].norm()+(row[2]-1).norm()+row[3].apply(x).norm()
    assert float(error) < TOL
    assert [isinstance(b, OperatorBlock) for b in row] == [True, False, False, True]


def test_source_seed_mixed_operator_identity():
    x = _x()
    result = ADChebfun(x).seed(3, [False, True, True, False])
    row = result.jacobian.blocks[0]
    error = row[0].norm()+row[1].apply(x).norm()+(row[2].apply(x)-x).norm()+row[3].norm()
    assert float(error) < TOL
    assert [isinstance(b, OperatorBlock) for b in row] == [False, True, True, False]


@pytest.mark.parametrize('kind', ['product', 'sum', 'quotient'])
def test_independent_two_function_derivatives(kind):
    domain = (-1., 1.)
    u = chebfun(lambda x: 2+x, domain=domain)
    v = chebfun(lambda x: 3-x, domain=domain)
    a, b = ADChebfun(u).seed(1, 2), ADChebfun(v).seed(2, 2)
    if kind == 'product':
        result, left, right = a*b, v, u
    elif kind == 'sum':
        result, left, right = a+b, chebfun(1.), chebfun(1.)
    else:
        result, left, right = a/b, 1/v, -u/v**2
    row = result.jacobian.blocks[0]
    p, q = chebfun(lambda x: .1+x*x), chebfun(lambda x: .2-x)
    assert float((row[0].apply(p)-left*p).norm(jnp.inf)) < 1e-12
    assert float((row[1].apply(q)-right*q).norm(jnp.inf)) < 1e-12
    assert result.is_linear == (kind == 'sum')


@pytest.mark.parametrize('operation', ['identity', 'cumsum', 'sum'])
def test_mixed_scalar_function_action_dimensions(operation):
    domain = (-1., 1.)
    u = chebfun(lambda x: 1+x, domain=domain)
    parameter = chebfun(2., domain=domain)
    flags = [True, False]
    a = ADChebfun(u).seed(1, flags)
    b = ADChebfun(parameter).seed(2, flags)
    result = a*b
    if operation != 'identity':
        result = getattr(result, operation)()
    matrix, rows = result.jacobian.matrix(12)
    disc = ChebColloc2Disc(12, domain)
    perturbation = jnp.concatenate([jnp.ones(12), jnp.asarray([3.])])
    expected = 2+3*(1+disc.points())
    if operation == 'cumsum':
        expected = 2*(disc.points()+1)+1.5*(disc.points()+1)**2
    elif operation == 'sum':
        expected = jnp.asarray([10.])
    assert matrix.shape == ((1 if operation == 'sum' else 12), 13)
    assert rows == [matrix.shape[0]]
    assert float(jnp.max(jnp.abs(matrix@perturbation-expected))) < 1e-12
    action = jax.jit(lambda values: matrix@values)
    actual, tangent = jax.jvp(action, (perturbation,), (perturbation,))
    assert float(jnp.max(jnp.abs(actual-expected))) < 1e-12
    assert float(jnp.max(jnp.abs(tangent-expected))) < 1e-12


def test_per_variable_linearity_and_reseed_reset():
    u = ADChebfun(chebfun(lambda x: 1+x)).seed(1, 3)
    v = ADChebfun(chebfun(lambda x: 2-x)).seed(2, 3)
    assert u.exp().linearity == (False, True, True)
    assert (2*u.exp()).linearity == (False, True, True)
    assert (u.exp()+v).linearity == (False, True, True)
    assert (u*v).linearity == (False, False, True)
    assert (u**0).linearity == (True, True, True)
    assert isinstance((u**0).jacobian, ChebMatrix)
    reset = u.exp().seed(3, 3)
    assert reset.linearity == (True, True, True) and reset.is_linear


def test_full_piecewise_seed_domain():
    domain = (-2., .5, 4.)
    u = ADChebfun(chebfun(lambda x: x, domain=domain)).seed(2, [False, True])
    assert u.domain == domain and u.jacobian.domain == domain
    assert float((u.jacobian.blocks[0][1].apply(u.func)-u.func).norm()) < TOL
