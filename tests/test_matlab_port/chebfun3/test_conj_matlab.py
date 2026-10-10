"""Original three predicates from tests/chebfun3/test_conj.m, Chebfun7574c77."""
from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 100 * ChebfunPref().cheb3Prefs.chebfun3eps


def test_native_conj_real():
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    h = f.conj()
    assert (f-h).norm() < TOL


def test_native_conj_imaginary():
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    h = (1j*f).conj()
    assert (1j*f+h).norm() < 100*TOL


def test_native_conj_complex():
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    g = chebfun3(lambda x, y, z: jnp.sin(x+y**2+z**3))
    h = (f+1j*g).conj()
    assert (f-1j*g-h).norm() < TOL
