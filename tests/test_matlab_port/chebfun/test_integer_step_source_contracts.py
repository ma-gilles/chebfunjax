"""Literal round clauses and breakpoint-value contracts.

Provenance
----------
MATLAB source: tests/chebfun/test_round.m, @chebfun/round.m; commit 7574c77.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = float(jnp.finfo(jnp.float64).eps)
X = jnp.asarray(json.loads((Path(__file__).parent / 'fixtures' /
                          'logical_source_matlab.json').read_text())['x'])


def native_round(x):
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + .5)


@pytest.mark.parametrize('clause', range(1, 7))
def test_original_round_clause(clause):
    if clause == 1:
        assert cj.chebfun().round().isempty()
        return
    domain = [-1., -.5, .5, 1.]
    if clause == 2:
        f = cj.chebfun(jnp.sin, domain=domain)
        exact = native_round(jnp.sin(X))
    elif clause in (3, 4, 5):
        f = cj.chebfun(jnp.exp, domain=domain)
        factor = {3: 1, 4: 1j, 5: 1+1j}[clause]
        f = factor*f
        exact = factor*native_round(jnp.exp(X))
    else:
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1),
                       domain=domain)
        exact = native_round(jnp.stack([jnp.sin(X), jnp.exp(X)], axis=-1))
    g = f.round()
    assert float(jnp.max(jnp.abs(g(X) - exact))) <= 10*float(g.vscale)*EPS


@pytest.mark.parametrize('value, expected', [(.5, 1.), (-.5, -1.), (2.5, 3.)])
def test_round_constant_ties(value, expected):
    f = cj.chebfun(lambda x: x*0 + value)
    g = f.round()
    assert bool(jnp.all(g.point_values == expected))
    assert bool(jnp.all(g(jnp.array([-.75, 0., .75])) == expected))


def test_round_inserted_point_values():
    g = cj.chebfun(lambda x: x).round()
    assert tuple(g.domain.breakpoints) == (-1., -.5, .5, 1.)
    assert bool(jnp.all(g.point_values == jnp.array([-1., -1., 1., 1.])))


def test_ceil_point_values_source_identity():
    f = cj.chebfun(lambda x: x, domain=[-1., 0., 1.])
    expected = -(-f).floor()
    actual = f.ceil()
    assert actual.domain == expected.domain
    assert bool(jnp.all(actual.point_values == expected.point_values))
    assert bool(jnp.all(actual(jnp.array([-.75, .25])) == jnp.array([0., 1.])))


def test_round_transpose():
    f = cj.chebfun(lambda x: x).T
    g = f.round()
    assert g.is_transposed
    assert bool(jnp.all(g.point_values == jnp.array([-1., -1., 1., 1.])))


def test_array_root_point_values_are_column_specific():
    f = cj.chebfun(lambda x: jnp.stack([x, 2*x], axis=-1))
    g = f.round()
    x = jnp.asarray(g.domain.breakpoints)
    expected = native_round(jnp.stack([x, 2*x], axis=-1))
    assert bool(jnp.all(g.point_values == expected))
