"""Literal six source clauses plus explicit all-output and chain controls.

Provenance: Chebfun 7574c77 tests/adchebfun/test_ellipj.m,
@adchebfun/valueTesting.m and taylorTesting.m. Six Taylor steps and three
outputs; source reduces convergence orders by row minimum. valueTesting's
omitted numOut defaults to one, despite the source test's all-output comment.
Seed6179 raw inputs were captured for the shared error-function fixture.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun


@pytest.fixture(scope='module')
def inputs():
    raw = json.loads(Path(__file__).with_name('erf_matlab_inputs.json').read_text())
    u, p = jnp.asarray(raw['raw_u']), jnp.asarray(raw['raw_p'])
    return .1*Chebfun.from_values(u)+.5, Chebfun.from_values(.1*u+.5), Chebfun.from_values(.01*p+.05)


@pytest.mark.parametrize('m', [.75, 1.])
def test_source_exact_value(inputs, m):
    u = inputs[0]
    assert float((u.ellipj(m)[0]-ADChebfun(u).ellipj(m)[0].func).norm(jnp.inf)) == 0


@pytest.mark.parametrize('m', [.75, 1.])
def test_source_six_step_three_output_taylor(inputs, m):
    _, u, p = inputs
    values = ADChebfun(u).ellipj(m)
    first, second = [], []
    for exponent in range(2, 8):
        perturbation = .2**exponent*p
        changed = ADChebfun(u+perturbation).ellipj(m)
        differences = [v.func-w.func for v, w in zip(changed, values)]
        first.append([d.norm(jnp.inf) for d in differences])
        second.append([(d-v.jacobian.apply(perturbation)).norm(jnp.inf)
                       for d, v in zip(differences, values)])
    for errors, expected in [(first, 1), (second, 2)]:
        orders = jnp.diff(jnp.log(jnp.asarray(errors)), axis=0)/jnp.log(.2)
        source_orders = jnp.min(orders, axis=1)
        assert float(jnp.max(jnp.abs(source_orders-expected))) < 1e-2


@pytest.mark.parametrize('m', [.75, 1.])
def test_source_nonlinearity(inputs, m):
    assert not ADChebfun(inputs[0]).ellipj(m)[0].is_linear


@pytest.mark.parametrize('m', [.75, 1.])
def test_all_three_values_and_nonlinearity(inputs, m):
    for expected, actual in zip(inputs[0].ellipj(m), ADChebfun(inputs[0]).ellipj(m)):
        assert float((expected-actual.func).norm(jnp.inf)) == 0
        assert not actual.is_linear


@pytest.mark.parametrize('m', [0., .75, 1.])
@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
def test_incoming_chain_domain(m, domain):
    u = chebfun(lambda x: .55+.002*x, domain=domain)
    h = chebfun(lambda x: .03+.001*x, domain=domain)
    argument = ADChebfun(u)**2
    sn, cn, dn = (u**2).ellipj(m)
    for output, derivative in zip(argument.ellipj(m), (cn*dn, -sn*dn, -m*sn*cn)):
        assert float((output.jacobian.apply(h)-derivative*(2*u*h)).norm(jnp.inf)) < 1e-12
        assert output.domain == argument.domain


def test_reject_parameter_ad(inputs):
    with pytest.raises(TypeError, match='first argument'):
        ADChebfun(inputs[0]).ellipj(ADChebfun(inputs[0]))
