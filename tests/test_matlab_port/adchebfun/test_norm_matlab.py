"""All15 original AD norm/testUnary clauses, with scalar-chain controls.

Provenance: tests/adchebfun/test_norm.m, @adchebfun/testUnary.m and
@adchebfun/taylorTesting.m, Chebfun7574c77680d7e82b79626300bf255498271a72df.
Original bounds: value50eps, Taylor slopes1e-2. Fixed degree7 polynomials
replace rand8 source inputs; no MATLAB RNG parity claim. Real norm inputs.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc, FunctionalBlock


@pytest.fixture(scope='module')
def data():
    u = chebfun(lambda x: .55+.02*(64*x**7-112*x**5+56*x**3-7*x)+.015*(4*x**3-3*x))
    p = chebfun(lambda x: .055+.003*(16*x**5-20*x**3+5*x)-.001*(2*x*x-1))
    return u, p


def _op(value, index):
    n = value.norm(2)
    if index == 0:
        return n
    if index == 1:
        return n**2
    if index == 2:
        return 1/n
    if index == 3:
        return 1/n**2
    # Source overwrites its first normFun5 assignment; this is the final one.
    return n**2+value.diff().norm()**3


@pytest.mark.parametrize('index', range(5))
def test_source_value_clause(data, index):
    u, _ = data
    assert abs(float(_op(ADChebfun(u), index).func-_op(u, index))) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('index', range(5))
def test_source_taylor_clause(data, index):
    u, p = data
    result = _op(ADChebfun(u), index)
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        delta = _op(ADChebfun(u+perturbation), index).func-result.func
        first.append(jnp.abs(delta))
        second.append(jnp.abs(delta-result.jacobian.apply(perturbation)))
    for errors, expected in [(first, 1), (second, 2)]:
        orders = jnp.diff(jnp.log(jnp.asarray(errors)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(orders-expected))) < 1e-2


@pytest.mark.parametrize('index', range(5))
def test_source_linearity_clause(data, index):
    assert not _op(ADChebfun(data[0]), index).is_linear


@pytest.mark.parametrize('index', range(5))
def test_scalar_chain_functional_shape_and_jax_action(data, index):
    u, p = data
    result = _op(ADChebfun(u), index)
    assert isinstance(result.jacobian, FunctionalBlock)
    disc = ChebColloc2Disc(32, (-1., 1.))
    row = result.jacobian.matrix(disc)
    assert row.shape == (32,)
    expected = result.jacobian.apply(p)
    action = jax.jit(lambda values: row@values)
    actual, tangent = jax.jvp(action, (p(disc.points()),), (p(disc.points()),))
    assert abs(float(actual-expected)) < 1e-12
    assert abs(float(tangent-expected)) < 1e-12


def test_source_non_l2_norm_is_primal(data):
    u, _ = data
    assert float(ADChebfun(u).norm(jnp.inf)) == float(u.norm(jnp.inf))
    assert float(ADChebfun(u).norm('fro').func) == float(u.norm(2))


def test_piecewise_norm_preserves_jacobian_domain():
    domain = (-1., .2, 1.)
    u = chebfun(1., domain=domain)
    result = ADChebfun(u).norm()
    assert result.domain == domain
    assert result.jacobian.domain == domain
    row = result.jacobian.matrix(12)
    assert row.shape == (24,)
    assert abs(float(row@jnp.ones(24)-jnp.sqrt(2.))) < 1e-12
