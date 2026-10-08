"""All original test_cumprodProd.m clauses through testUnary.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
tests/adchebfun/test_cumprodProd.m and @adchebfun/testUnary.m/taylorTesting.m.
Original50eps primal and1e-2 Taylor bounds, function infinity norms.
Inputs captured from MATLAB R2025b seedRNG(6179), pinned source7574c77.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

OPERATIONS = ['cumprod', 'prod']


@pytest.fixture(scope='module')
def inputs():
    capture = json.loads(Path(__file__).with_name('erf_matlab_inputs.json').read_text())
    raw_u, raw_p = jnp.asarray(capture['raw_u']), jnp.asarray(capture['raw_p'])
    value_u = .1*Chebfun.from_values(raw_u)+.5
    taylor_u = Chebfun.from_values(.1*raw_u+.5)
    p = Chebfun.from_values(.01*raw_p+.05)
    return value_u, taylor_u, p


def _norm(value):
    return float(value.norm(jnp.inf)) if isinstance(value, Chebfun) else float(jnp.abs(value))


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_value_clause(inputs, operation):
    u, _, _ = inputs
    expected = getattr(u, operation)()
    actual = getattr(ADChebfun(u), operation)().func
    assert _norm(expected-actual) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_taylor_clause(inputs, operation):
    _, u, p = inputs
    result = getattr(ADChebfun(u), operation)()
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        changed = getattr(ADChebfun(u+perturbation), operation)()
        delta = changed.func-result.func
        action = result.jacobian.apply(perturbation)
        first.append(_norm(delta))
        second.append(_norm(delta-action))
    for errors, expected in [(first, 1), (second, 2)]:
        orders = jnp.diff(jnp.log(jnp.asarray(errors)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(orders-expected))) < 1e-2


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_linearity_clause(inputs, operation):
    assert not getattr(ADChebfun(inputs[0]), operation)().is_linear


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
def test_constant_integrand_chain_and_anchor(operation, domain):
    u = chebfun(lambda x: 2.+0*x, domain=domain)
    h = chebfun(lambda x: 1.+0*x, domain=domain)
    result = getattr(ADChebfun(u)**2, operation)()
    action = result.jacobian.apply(h)
    a, b = domain[0], domain[-1]
    if operation == 'cumprod':
        points = jnp.linspace(a, b, 17)
        assert float(jnp.max(jnp.abs(result.func(points)-4.**(points-a)))) < 1e-10
        assert float(jnp.max(jnp.abs(action(points)-(points-a)*4.**(points-a)))) < 1e-10
        assert abs(float(result.func(a))-1.) < 1e-13
    else:
        assert abs(float(result.func)-4.**(b-a)) < 1e-10
        assert abs(float(action)-(b-a)*4.**(b-a)) < 1e-10
    assert result.domain == domain
    assert not result.is_linear
