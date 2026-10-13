"""Native rdivide error predicates13 (both assignments),14 and15.

Provenance
----------
MATLAB source : tests/chebfun/test_rdivide.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj

PREFIX = 'CHEBFUN:CHEBFUN:rdivide:columnRdivide:'
DOM = (-1., -.5, 0., .5, 1.)


def test_source_13_numeric_zero():
    f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1),
                   domain=DOM)
    with pytest.raises(ValueError, match=PREFIX + 'divisionByZero:'):
        f / 0


def test_source_13_zero_chebfun_overwrite():
    f = cj.chebfun(lambda x: 1 + 0*x)
    g = cj.chebfun(lambda x: 0*x)
    with pytest.raises(ValueError, match=PREFIX + 'divisionByZeroChebfun:'):
        f / g


def test_source_14_orientation_error():
    f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1),
                   domain=DOM)
    g = cj.chebfun(lambda x: jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1),
                   domain=(-1., 1.))
    with pytest.raises(ValueError, match=PREFIX + 'dim:'):
        f / g.T


def test_source_15_domain_error():
    f = cj.chebfun(jnp.exp, domain=(-1., 1.))
    g = cj.chebfun(jnp.exp, domain=(0., 2.))
    with pytest.raises(ValueError, match=PREFIX + 'domain:'):
        f / g
