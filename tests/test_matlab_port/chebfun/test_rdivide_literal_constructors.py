"""Restore actual constructors in native rdivide predicates8–12 and18.

Provenance: tests/chebfun/test_rdivide.m, Chebfun7574c77.
Existing private-conversion and callable controls remain separate. The native
source has26 final pass slots and28 assigned expressions; this file restores
constructor operands only and does not replace overwritten24's strict xfail.
"""
import json
import math
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = jnp.finfo(jnp.float64).eps
DOM = (-1., -.5, 0., .5, 1.)
X = jnp.asarray(json.loads((Path(__file__).parent/'fixtures/abs_seed6178_native.json').read_text())['x'])


def scale(f):
    v = f.vscale
    return jnp.max(v() if callable(v) else v)


def positive_negative_exp(x):
    return jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1)


def negative_positive_exp(x):
    return jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1)


@pytest.mark.parametrize('clause', range(8, 13))
def test_actual_quasimatrix_constructor(clause):
    f = cj.chebfun(positive_negative_exp, domain=DOM)
    fq = cj.quasimatrix(positive_negative_exp, domain=DOM)
    if clause == 8:
        h = 1/fq
        expected = negative_positive_exp(X)
    else:
        g = cj.chebfun(negative_positive_exp, domain=(-1., 1.))
        gq = cj.quasimatrix(negative_positive_exp, domain=(-1., 1.))
        expected = jnp.stack((jnp.exp(2*X), jnp.exp(-2*X)), axis=-1)
        if clause == 9:
            h = fq/g
        elif clause == 10:
            h = f/gq
        elif clause == 11:
            h = fq/gq
        else:
            h = fq.T/gq.T
            expected = expected.T
    error = jnp.max(jnp.abs(h(X)-expected))
    bound = 10*EPS*scale(h)
    print('clause',clause,'error',float(error),'bound',float(bound))
    assert error < bound


def test_literal_18_periodic_bernoulli():
    k = 10
    z = cj.chebfun('4*exp(1i*s)', domain=(0.,2*math.pi), trig=True)
    f = z/(z.exp()-1)
    b10 = math.factorial(k)*((f/z**(k+1))*z.diff()).sum()/(2j*math.pi)
    error = jnp.abs(5/66-b10)
    bound = 100*EPS*scale(f)
    print('clause18','error',error.tolist(),'bound',float(bound))
    assert error < bound
