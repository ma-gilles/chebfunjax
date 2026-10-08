"""All three original source AD volt predicates with seed6179 inputs.

Provenance
----------
MATLAB source: tests/adchebfun/test_volt.m and @adchebfun/testUnary.m.
Chebfun commit: 7574c77
Existing native seed6179 raw inputs match valueTesting/taylorTesting.
No fresh native Fredholm/Volterra execution is claimed.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun


def kernel(s, t):
    return jnp.exp(-(s-t)**2)


@pytest.fixture(scope='module')
def inputs():
    data = json.loads(Path(__file__).with_name('erf_matlab_inputs.json').read_text())
    raw_u, raw_p = jnp.asarray(data['raw_u']), jnp.asarray(data['raw_p'])
    return (.1*Chebfun.from_values(raw_u)+.5,
            Chebfun.from_values(.1*raw_u+.5),
            Chebfun.from_values(.01*raw_p+.05))


def test_source_value_clause(inputs):
    u, _, _ = inputs
    expected = cj.volt(kernel, u)
    actual = cj.volt(kernel, ADChebfun(u)).func
    assert float((expected-actual).norm(jnp.inf)) == 0


def test_source_taylor_clause(inputs):
    _, u, p = inputs
    result = cj.volt(kernel, ADChebfun(u))
    first, second = [], []
    for exponent in range(2, 4):
        perturbation = .2**exponent*p
        changed = cj.volt(kernel, ADChebfun(u+perturbation))
        delta = changed.func-result.func
        action = result.jacobian.apply(perturbation)
        first.append(delta.norm(jnp.inf))
        second.append((delta-action).norm(jnp.inf))
    orders = jnp.diff(jnp.log(jnp.asarray(first)))/jnp.log(.2)
    assert float(jnp.max(jnp.abs(orders-1))) < 1e-2
    assert float(jnp.max(jnp.abs(jnp.asarray(second)))) < 1e-12


def test_source_linearity_clause(inputs):
    assert cj.volt(kernel, ADChebfun(inputs[0])).is_linear
