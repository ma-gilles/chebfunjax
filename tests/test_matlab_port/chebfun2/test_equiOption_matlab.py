"""Port of MATLAB Chebfun tests/chebfun2/test_equiOption.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_equiOption.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


def _case(dom, f, tol):
    x = np.linspace(dom[0], dom[1], 100)
    y = np.linspace(dom[2], dom[3], 100)
    xx, yy = np.meshgrid(x, y)
    A = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy)))
    g = chebfun2(A, domain=dom, equi=True)
    h = chebfun2(f, domain=dom)
    return float((h - g).norm()) < tol


class TestChebfun2EquiOption:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        assert _case((-1, 1, -1, 1), lambda x, y: jnp.cos(x + y), tol)       # pass(1)
        assert _case((-1, 2, -2, 1), lambda x, y: jnp.cos(x + y), tol)       # pass(2)
        assert _case((-1, 2, -2, 1), lambda x, y: jnp.cos(x + 2 * y), tol)   # pass(3)
        h = 1e-3
        assert _case((1 - h, 1 + h, 1 - 2 * h, 1 + 2 * h),
                     lambda x, y: jnp.cos(x + 2 * y), tol)                  # pass(4)
