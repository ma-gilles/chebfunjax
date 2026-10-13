"""Native rdivide predicates 1–7 and 25–26, plus one supplemental control.

Provenance
----------
MATLAB source : tests/chebfun/test_rdivide.m
Chebfun commit: 7574c77

The native first draw at seed6178 is shared with test_abs.m. Later native
quasimatrix, remaining error, singular, unbounded and periodic clauses
remain unported here; these nine predicates do not establish full coverage.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = float(jnp.finfo(jnp.float64).eps)
X = jnp.asarray(json.loads(
    (Path(__file__).parent / 'fixtures' / 'abs_seed6178_native.json').read_text()
)['x'])
DOM = (-1., -.5, 0., .5, 1.)


def test_source_01_empty():
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun()
    assert (f / g).isempty() and (g / f).isempty()


def test_source_02_zero_numerator():
    f = cj.chebfun(jnp.sin)
    assert (0. / f).iszero()


@pytest.mark.parametrize('clause', range(3, 8))
def test_source_03_07_scalar_array(clause):
    if clause < 5:
        f = cj.chebfun(jnp.exp, domain=DOM)
        if clause == 3:
            h = 1. / f
            expected = jnp.exp(-X)
        else:
            g = cj.chebfun(lambda x: jnp.exp(-x), domain=(-1., 1.))
            h = f / g
            expected = jnp.exp(2 * X)
    else:
        f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1),
                       domain=DOM)
        if clause == 5:
            h = 1. / f
            expected = jnp.stack((jnp.exp(-X), jnp.exp(X)), axis=-1)
        else:
            g = cj.chebfun(lambda x: jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1),
                           domain=(-1., 1.))
            h = f.T / g.T if clause == 7 else f / g
            expected = jnp.stack((jnp.exp(2 * X), jnp.exp(-2 * X)), axis=-1)
            if clause == 7:
                expected = expected.T
    assert jnp.max(jnp.abs(h(X) - expected)) < 10 * jnp.max(h.vscale) * EPS


def test_supplemental_sine_over_shifted_cosine():
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun(lambda x: 2 + jnp.cos(x))
    h = f / g
    assert jnp.max(jnp.abs(h(X) - jnp.sin(X) / (2 + jnp.cos(X)))) < 1e3 * EPS


@pytest.mark.parametrize('denominator', ((0., 0.), (1., 0., 1.)))
def test_source_25_26_numeric_zero_columns(denominator):
    if len(denominator) == 2:
        f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1))
    else:
        f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x), jnp.sin(x)), axis=-1))
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZero'):
        f / jnp.asarray(denominator)
