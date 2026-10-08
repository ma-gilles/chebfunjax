"""All original test_pow2Sqrt.m clauses through testUnary.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
tests/adchebfun/test_pow2Sqrt.m and @adchebfun/testUnary.m/taylorTesting.m.
Original50eps primal and1e-2 Taylor bounds, function infinity norms.
Fixed degree7 polynomials replace rand8; no RNG-stream parity claimed.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun

OPERATIONS = ['pow2', 'sqrt']


@pytest.fixture(scope='module')
def inputs():
    u = chebfun(lambda x: .55+.012*x+.008*x**2+.003*x**4+.001*x**7)
    p = chebfun(lambda x: .055+.001*x+.0008*x**2+.0003*x**4+.0001*x**7)
    return u, p


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_value_clause(inputs, operation):
    u, _ = inputs
    expected = getattr(u, operation)()
    actual = getattr(ADChebfun(u), operation)().func
    assert abs(float((expected-actual).norm(jnp.inf))) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_taylor_clause(inputs, operation):
    u, p = inputs
    result = getattr(ADChebfun(u), operation)()
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        changed = getattr(ADChebfun(u+perturbation), operation)()
        delta = changed.func-result.func
        action = result.jacobian.apply(perturbation)
        first.append(delta.norm(jnp.inf))
        second.append((delta-action).norm(jnp.inf))
    for errors, expected in [(first, 1), (second, 2)]:
        orders = jnp.diff(jnp.log(jnp.asarray(errors)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(orders-expected))) < 1e-2


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_linearity_clause(inputs, operation):
    assert not getattr(ADChebfun(inputs[0]), operation)().is_linear
