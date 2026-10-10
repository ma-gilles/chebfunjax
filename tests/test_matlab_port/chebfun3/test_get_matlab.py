"""All five native test_get.m predicates, Chebfun commit7574c77.

Explicit dot records represent native property subsref. Tucker's Python
factor lists are assembled into the same continuous Chebfun panels.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_get_five_predicates():
    tol = 1e2 * ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    core, cols, rows, tubes = f.tucker()

    def prop(name):
        return f.subsref({'type': '.', 'subs': name})

    assert jnp.linalg.norm(jnp.asarray([-1, 1, -1, 1, -1, 1])-prop('domain').reshape(-1)) < tol
    assert jnp.linalg.norm(core.reshape(-1, order='F')-prop('core').reshape(-1, order='F')) < tol
    assert (Chebfun.horzcat(*cols)-prop('cols')).norm() < tol
    assert (Chebfun.horzcat(*rows)-prop('rows')).norm() < tol
    assert (Chebfun.horzcat(*tubes)-prop('tubes')).norm() < tol
