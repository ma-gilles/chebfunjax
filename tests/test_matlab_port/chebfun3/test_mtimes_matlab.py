"""All four original tests/chebfun3/test_mtimes.m predicates, Chebfun7574c77.

Python @ selects matrix multiplication; * selects elementwise multiplication.
Original domain, functions, continuous norms and1000prefeps retained.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1000*ChebfunPref().cheb3Prefs.chebfun3eps
DOMAIN = (-1., 1., 0., 2., -4., -2.)


@pytest.fixture(scope='module')
def field():
    return chebfun3(lambda x, y, z: x+jnp.cos(y+2*z), DOMAIN)


def test_native_mtimes_scalar(field):
    c = 10
    h = field @ c
    exact = chebfun3(lambda x, y, z: c*(x+jnp.cos(y+2*z)), DOMAIN)
    assert (exact-h).norm() < TOL


def test_native_mtimes_chebfun(field):
    g = chebfun(lambda x: 2*x+3)
    h = field @ g
    exact = Chebfun2.from_function(lambda y, z: 4/3+6*jnp.cos(y+2*z), domain=DOMAIN[2:])
    assert (exact-h).norm() < TOL


def test_native_mtimes_chebfun2(field):
    g = Chebfun2.from_function(lambda x, t: x+t, domain=(-1., 1., 3., 5.))
    h = field @ g
    exact = chebfun3(lambda t, y, z: 2/3+2*t*jnp.cos(y+2*z), (3., 5.)+DOMAIN[2:])
    assert (exact-h).norm() < TOL


def test_native_mtimes_chebfun3_error(field):
    g = chebfun3(lambda x, y, z: x+y+z)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:mtimes'):
        field @ g
