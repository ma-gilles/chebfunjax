"""All original test_airy.m clauses through testUnary.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
tests/adchebfun/test_airy.m and @adchebfun/testUnary.m/taylorTesting.m.
Original50eps primal and1e-2 Taylor bounds, function infinity norms.
Inputs captured from MATLAB R2025b seedRNG(6179), pinned source7574c77.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

OPERATIONS = [0, 2]


@pytest.fixture(scope='module')
def inputs():
    capture = json.loads(Path(__file__).with_name('erf_matlab_inputs.json').read_text())
    raw_u, raw_p = jnp.asarray(capture['raw_u']), jnp.asarray(capture['raw_p'])
    value_u = .1*Chebfun.from_values(raw_u)+.5
    taylor_u = Chebfun.from_values(.1*raw_u+.5)
    p = Chebfun.from_values(.01*raw_p+.05)
    return value_u, taylor_u, p


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_value_clause(inputs, operation):
    u, _, _ = inputs
    expected = u.airy(operation)
    actual = ADChebfun(u).airy(operation).func
    assert abs(float((expected-actual).norm(jnp.inf))) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_taylor_clause(inputs, operation):
    _, u, p = inputs
    result = ADChebfun(u).airy(operation)
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        changed = ADChebfun(u+perturbation).airy(operation)
        delta = changed.func-result.func
        action = result.jacobian.apply(perturbation)
        first.append(delta.norm(jnp.inf))
        second.append((delta-action).norm(jnp.inf))
    for errors, expected in [(first, 1), (second, 2)]:
        orders = jnp.diff(jnp.log(jnp.asarray(errors)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(orders-expected))) < 1e-2


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_linearity_clause(inputs, operation):
    assert not ADChebfun(inputs[0]).airy(operation).is_linear


@pytest.mark.parametrize('kind', [0, 2])
@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
def test_source_multiplier_preserves_incoming_chain_and_domain(kind, domain):
    u = chebfun(lambda x: .55+.002*x, domain=domain)
    h = chebfun(lambda x: .03+.001*x, domain=domain)
    argument = ADChebfun(u)**2
    result = argument.airy(kind)
    expected = (u**2).airy(kind+1)*(2*u*h)
    assert float((result.jacobian.apply(h)-expected).norm(jnp.inf)) < 1e-12
    assert result.domain == argument.domain
    assert not result.is_linear


def test_default_dispatch():
    u = chebfun(lambda x: .55+.002*x)
    assert float((ADChebfun(u).airy().func-u.airy(0)).norm(jnp.inf)) < 50*jnp.finfo(jnp.float64).eps
