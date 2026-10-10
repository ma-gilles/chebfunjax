"""Literal three native test_tucker.m predicates, Chebfun7574c77.

The two source function sections run separately; predicate3 retains the
second function and its previously returned core. Native linspace defaults
use100 points. Python Tucker lists are assembled as continuous panels.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref


def panel_values(columns, points):
    return jnp.asarray(Chebfun.horzcat(*columns)(points)).reshape((points.size, -1))


def test_native_tucker_unit_cube():
    tol = 1000*ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    core, cols, rows, tubes = f.tucker()
    print({'core_shape': list(core.shape), 'factor_lengths': list(f.length())}, flush=True)
    x = jnp.linspace(-1., 1., 100)
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing='ij')
    fvals = f(xx, yy, zz)
    stval = Chebfun3.txm(Chebfun3.txm(Chebfun3.txm(core, panel_values(cols, x), 1),
                                   panel_values(rows, x), 2), panel_values(tubes, x), 3)
    err = jnp.linalg.norm(stval.reshape(-1, order='F')-fvals.reshape(-1, order='F'))
    assert err < tol


def test_native_tucker_box_and_core():
    tol = 1000*ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z), (-3., 4., -1., 3., 2., 4.))
    fcore, cols, rows, tubes = f.tucker()
    print({'core_shape': list(fcore.shape), 'factor_lengths': list(f.length())}, flush=True)
    x = jnp.linspace(-3., 4., 100)
    y = jnp.linspace(-1., 3., 100)
    z = jnp.linspace(2., 4., 100)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
    fvals = f(xx, yy, zz)
    stval = Chebfun3.txm(Chebfun3.txm(Chebfun3.txm(fcore, panel_values(cols, x), 1),
                                   panel_values(rows, y), 2), panel_values(tubes, z), 3)
    err = jnp.linalg.norm(stval.reshape(-1, order='F')-fvals.reshape(-1, order='F'))
    assert err < tol
    core = f.tucker()[0]
    assert jnp.linalg.norm(fcore.reshape(-1, order='F')-core.reshape(-1, order='F')) < tol
