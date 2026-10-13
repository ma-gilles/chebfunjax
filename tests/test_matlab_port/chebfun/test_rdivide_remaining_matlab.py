"""Remaining native division predicates, retaining overwritten pass24.

Provenance
----------
MATLAB source: tests/chebfun/test_rdivide.m
Chebfun commit: 7574c77
"""
import math

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = jnp.finfo(jnp.float64).eps
DOM = (0., math.pi, 2*math.pi)


def test_source_18_periodic_bernoulli():
    z = cj.chebfun(lambda s: 4*jnp.exp(1j*s), domain=(0., 2*math.pi), trig=True)
    f = z/(z.exp()-1)
    b10 = math.factorial(10)*((f/z**11)*z.diff()).sum()/(2j*math.pi)
    assert jnp.abs(5/66-b10) < 100*EPS*f.vscale


def test_source_19_20_mixed_scalar():
    f = cj.chebfun(lambda x: x+x**2, domain=DOM)
    g = cj.chebfun(lambda x: 2+jnp.cos(x), domain=(0.,2*math.pi), trig=True)
    h1 = f/g
    assert type(h1.funs[0].tech) is type(f.funs[0].tech)
    h2 = cj.chebfun(lambda x: (x+x**2)/(2+jnp.cos(x)), domain=DOM)
    assert (h1-h2).norm(jnp.inf) < 100*EPS*h2.vscale


def test_source_21_23_mixed_array():
    f = cj.chebfun(lambda x: jnp.stack((jnp.cos(x),jnp.sin(x)),axis=-1), domain=(0.,2*math.pi), trig=True)
    g = cj.chebfun(lambda x: jnp.stack((2+x,2+x**3),axis=-1), domain=DOM)
    h1 = f/g
    for k in range(2):
        assert type(h1[:,k].funs[0].tech) is type(g[:,k].funs[0].tech)
    h2 = cj.chebfun(lambda x: jnp.stack((jnp.cos(x)/(2+x),jnp.sin(x)/(2+x**3)),axis=-1), domain=DOM)
    assert jnp.all((h1-h2).norm(jnp.inf) < 100*EPS*h2.vscale)


@pytest.fixture(scope="module")
def removable_quotients():
    y1 = cj.chebfun('abs(sin(5*x))', splitting=True)
    y2 = cj.chebfun('abs(sin(4*x))', splitting=True)
    y3 = y2-y1
    g = ((2*y3).exp()-y3.exp())/y3
    mg = g.merge()
    return g, mg


@pytest.mark.xfail(strict=True, reason=(
    "Pinned MATLAB7574c77/R2025b also fails the overwritten pass24: "
    "g has10 pieces; mg=merge(g) has6. See native rdivide capture."))
def test_source_24_removable_zeros_overwritten(removable_quotients):
    g, _ = removable_quotients
    assert jnp.all(g.vscale < jnp.inf) and len(g.funs) == 6


def test_native_captured_merge_control(removable_quotients):
    # Independent captured-native metadata, not a replacement for pass24.
    g, mg = removable_quotients
    assert len(g.funs) == 10
    assert len(mg.funs) == 6


def test_source_24_numeric_zero_error():
    f = cj.chebfun(jnp.exp)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZero'):
        f/0
