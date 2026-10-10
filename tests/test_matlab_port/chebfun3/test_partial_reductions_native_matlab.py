"""All original sum, sum2, mean and mean2 assertion slots.

Provenance
----------
MATLAB source: tests/chebfun3/test_{sum,sum2,mean,mean2}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Existing additional Python fixtures remain in their original files.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d import chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

EPS = ChebfunPref().cheb3Prefs.chebfun3eps


@pytest.mark.parametrize('slot', range(1, 14))
def test_native_sum_all13(slot):
    if slot <= 4:
        dom = (11, 14, 7, 10, 5, 9)
        def fn(x, y, z):
            return x**2 + 4*y - 5*z
        dim = (1, 1, 2, 3)[slot-1]
        exact = (lambda y, z: 12*y - 15*z + 471,
                 lambda y, z: 12*y - 15*z + 471,
                 lambda x, z: 3*x**2 - 15*z + 102,
                 lambda x, y: 4*x**2 + 16*y - 140)[slot-1]
    else:
        dom = (-2, 3, -2, 2, -5, -3)
        fn = (lambda x, y, z: jnp.cos(y) + jnp.sin(z),
              lambda x, y, z: jnp.cos(x) + jnp.sin(z),
              lambda x, y, z: jnp.cos(x) + jnp.sin(y))[(slot-5) % 3]
        dim = 1 + (slot-5)//3
        exact = (
            lambda y, z: 5*(jnp.cos(y) + jnp.sin(z)),
            lambda y, z: jnp.sin(3.) + jnp.sin(2.) + 5*jnp.sin(z),
            lambda y, z: jnp.sin(3.) + jnp.sin(2.) + 5*jnp.sin(y),
            lambda x, z: 2*jnp.sin(2.) + 4*jnp.sin(z),
            lambda x, z: 4*(jnp.cos(x) + jnp.sin(z)),
            lambda x, z: 4*jnp.cos(x),
            lambda x, y: 2*jnp.cos(y) + jnp.cos(5.) - jnp.cos(3.),
            lambda x, y: 2*jnp.cos(x) + jnp.cos(5.) - jnp.cos(3.),
            lambda x, y: 2*(jnp.cos(x) + jnp.sin(y)),
        )[slot-5]
    survivor_domain = tuple(v for i in range(3) if i != dim-1
                            for v in dom[2*i:2*i+2])
    f = Chebfun3.from_function(fn, domain=dom)
    actual = f.sum() if slot in (1, 5, 6, 7) else f.sum(dim)
    reference = chebfun2(exact, domain=survivor_domain)
    assert (actual - reference).norm() < 1e4*EPS


@pytest.mark.parametrize('slot', range(1, 11))
def test_native_sum2_all10(slot):
    if slot <= 7:
        dom = (11, 14, 7, 10, 5, 9)
        f = Chebfun3.from_function(lambda x, y, z: x**2 + 4*y - 5*z,
                                  domain=dom)
        dims = ((1, 2), (1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1))[slot-1]
        if slot <= 3:
            def fn(z):
                return 1719 - 45*z
        elif slot <= 5:
            def fn(x):
                return 12*x**2 - 12
        else:
            def fn(y):
                return 48*y + 1464
    else:
        dom = (-1, 1, -1, 1, -1, 1)
        f = Chebfun3.from_function(lambda x, y, z: 1j*x**2 + 2j*y**2 + z**2)
        dims = ((1, 2), (1, 3), (2, 3))[slot-8]
        fn = (lambda z: 4*z**2 + 4j,
              lambda y: 8j*y**2 + (4/3)*(1+1j),
              lambda x: 1j*(4*x**2 + 8/3) + 4/3)[slot-8]
    survivor = next(i for i in (1, 2, 3) if i not in dims)
    reference = chebfun(fn, domain=dom[2*(survivor-1):2*survivor])
    actual = f.sum2() if slot in (1, 8) else f.sum2(dims)
    assert (actual - reference).norm() < 1e5*EPS


def test_native_mean_original():
    f = Chebfun3.from_function(lambda x, y, z: x**2*y**4*jnp.exp(z))
    reference = chebfun2(lambda y, z: y**4*jnp.exp(z)/3)
    assert (f.mean() - reference).norm() < EPS


def test_native_mean2_original():
    f = Chebfun3.from_function(lambda x, y, z: x**2*y**4*jnp.exp(z))
    reference = chebfun(lambda z: jnp.exp(z)/15)
    assert (f.mean2() - reference).norm() < EPS
