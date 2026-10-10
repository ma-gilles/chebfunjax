"""Original deterministic Diskfun SVD3 and norm3 assertions.

Provenance
----------
MATLAB source : tests/diskfun/test_svd.m, tests/diskfun/test_norm.m
Chebfun commit: 7574c77
Cartesian functions use the public polar-coordinate adapter, unchanged bounds.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps


@pytest.fixture(scope='module')
def svd_field():
    f = Diskfun.from_function(lambda t, r: jnp.cos((r*jnp.cos(t))*(r*jnp.sin(t))**2))
    return f, f.svd()


def test_native_svd_norm_identity(svd_field):
    f, s = svd_field
    assert abs(f.norm()**2-jnp.sum(s**2)) < TOL


def test_native_svd_resolved_tail(svd_field):
    _, s = svd_field
    assert s[-1] < 10*TOL


def test_native_svd_scale_invariance(svd_field):
    f, s = svd_field
    t = (100*f).svd()
    assert jnp.linalg.norm(s-t/100) < TOL


def test_native_norm_constant():
    f = Diskfun.from_function(lambda t, r: 1+0*r*jnp.cos(t))
    assert abs(jnp.sqrt(f.sum2())-f.norm()) < TOL


def test_native_norm_cosxy():
    f = Diskfun.from_function(lambda t, r: jnp.cos((r*jnp.cos(t))*(r*jnp.sin(t))))
    s = f.svd()
    assert abs(jnp.sum(s**2)-f.norm()**2) < TOL


def test_native_norm_inf_linear():
    f = Diskfun.from_function(lambda t, r: r*jnp.cos(t)+r*jnp.sin(t))
    assert abs(f.norm(jnp.inf)-jnp.sqrt(2)) < TOL
