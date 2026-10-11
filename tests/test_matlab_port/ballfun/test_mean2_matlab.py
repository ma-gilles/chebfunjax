"""Port of MATLAB Chebfun tests/ballfun/test_mean2.m (Fable 5).

FIXED: Ballfun.mean2(dims) added in the Fable 5 audit (average over
two spherical coordinates -> 1D Chebfun in the survivor; trig for
the lambda survivor).

Provenance
----------
MATLAB source : tests/ballfun/test_mean2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

TOL = 1e-11
TS = jnp.asarray(np.linspace(0.05, 3.1, 9))
RS = jnp.asarray(np.linspace(0.0, 1.0, 9))


class TestBallfunMean2:
    def test_constant_means(self):
        # pass(1)-(2)
        f1 = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        assert float(jnp.max(jnp.abs(
            f1.mean2((1, 2))(TS) - 1.0))) < TOL
        f2 = Ballfun.from_function(lambda x, y, z: 2.0 + 0 * x)
        assert float(jnp.max(jnp.abs(
            f2.mean2((2, 3))(RS) - 2.0))) < TOL

    def test_radial_profile(self):
        f = Ballfun.from_function(lambda r, lam, th: r ** 2,
                                  spherical=True)
        assert float(jnp.max(jnp.abs(
            f.mean2((2, 3))(RS) - RS ** 2))) < TOL


import pytest

from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.mark.parametrize("dims,value", [((1,2),1),((2,3),2),((1,3),3)])
def test_native_mean2_norm(dims,value):
    f = Ballfun.from_function(lambda x,y,z: value)
    if dims == (2,3):
        exact = chebfun(lambda x: value+0*x)
    else:
        exact = chebfun(lambda x: value+0*x, domain=(-np.pi,np.pi), trig=True)
    assert (f.mean2(dims)-exact).norm() < 1e4*np.finfo(float).eps


@pytest.mark.parametrize("dims", [(1,2),(1,3),(2,3)])
def test_complex_weighted_mean2(dims):
    f = Ballfun.from_function(lambda r,l,t: (1+2j)*r*r, spherical=True)
    got = f.mean2(dims)
    x = jnp.array([-.7,.2,.8])
    expected = (1+2j)*(x*x if dims == (2,3) else 3/5)
    assert jnp.max(jnp.abs(got(x)-expected)) < 1e4*np.finfo(float).eps


def test_default_mean2_is_angular():
    f = Ballfun.from_function(lambda r,l,t: r*r, spherical=True)
    x = jnp.array([-.7,.2,.8])
    assert jnp.max(jnp.abs(f.mean2()(x)-x*x)) < 1e4*np.finfo(float).eps
