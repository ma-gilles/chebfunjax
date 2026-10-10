"""Native unary constructor, sign and principal-value source controls.

MATLAB source: @chebfun3/{exp,cos,tanh,abs,sqrt,log}.m,7574c77.
"""
import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1000*ChebfunPref().cheb3Prefs.chebfun3eps
DOMAIN = (-math.pi, math.pi)*3
POINTS = jnp.asarray([-2., -.5, 1.7])


@pytest.mark.parametrize('name', ['exp', 'cos', 'tanh'])
def test_periodic_input_uses_default_constructor(name):
    f = chebfun3(lambda x, y, z: .2*jnp.cos(x), DOMAIN, trig=True)
    g = getattr(f, name)()
    assert f.isPeriodicTech() and not g.isPeriodicTech()
    assert tuple(g.domain) == DOMAIN
    zero = jnp.zeros_like(POINTS)
    expected = getattr(jnp, name)(.2*jnp.cos(POINTS))
    assert jnp.max(jnp.abs(g(POINTS, zero, zero)-expected)) < TOL


@pytest.mark.parametrize('name', ['sqrt', 'log', 'abs'])
def test_real_sign_change_preserves_native_error(name):
    f = chebfun3(lambda x, y, z: x)
    with pytest.raises(ValueError, match=f'CHEBFUN:CHEBFUN3:{name}:notSmooth'):
        getattr(f, name)()


@pytest.mark.parametrize('name', ['sqrt', 'log'])
def test_negative_periodic_input_principal_complex_and_technology(name):
    f = chebfun3(lambda x, y, z: -4+.2*jnp.cos(x), DOMAIN, trig=True)
    g = getattr(f, name)()
    assert g.isPeriodicTech()
    zero = jnp.zeros_like(POINTS)
    expected = getattr(jnp, name)((-4+.2*jnp.cos(POINTS)).astype(jnp.complex128))
    assert jnp.max(jnp.abs(g(POINTS, zero, zero)-expected)) < TOL


def test_abs_positive_identity_and_complex_composition():
    f = chebfun3(lambda x, y, z: 2+.1*x)
    assert f.abs() is f
    h = chebfun3(lambda x, y, z: (1+1j)*(2+.1*x))
    x = jnp.asarray([-.7, .1, .6])
    assert jnp.max(jnp.abs(h.abs()(x, 0*x, 0*x)-jnp.sqrt(2.)*(2+.1*x))) < TOL


@pytest.mark.parametrize('name', ['exp', 'cos', 'tanh', 'sqrt', 'log', 'abs'])
def test_empty_unary(name):
    assert getattr(chebfun3(), name)().isempty()
