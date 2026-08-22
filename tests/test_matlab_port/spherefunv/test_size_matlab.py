"""Port of MATLAB Chebfun tests/spherefunv/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)


class TestSpherefunvSize:
    def test_all_matlab_assertions(self):
        F = Spherefunv(_sph(lambda x, y, z: jnp.cos(x)),
                       _sph(lambda x, y, z: jnp.sin(y)),
                       _sph(lambda x, y, z: jnp.cos(1 + z)))
        assert F.size == (3, np.inf, np.inf)
        assert F.size[0] == 3
        assert F.size[1] == np.inf
        assert F.size[2] == np.inf
