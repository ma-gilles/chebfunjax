"""Port of MATLAB Chebfun tests/spherefun/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefunv import Spherefunv

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


class TestSpherefunCurl:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        u = sph_xyz(lambda x, y, z: x * y * z).curl()
        assert isinstance(u, Spherefunv)                                     # pass(1)
        f = sph_xyz(lambda x, y, z: 0 * x)
        assert float(f.curl().norm()) < tol                                  # pass(2)

        u = sph_xyz(lambda x, y, z: (x - y) * z).curl()
        exact = Spherefunv(sph_xyz(lambda x, y, z: (x - y) * y + z ** 2),
                           sph_xyz(lambda x, y, z: (y - x) * x + z ** 2),
                           sph_xyz(lambda x, y, z: -(x + y) * z))
        assert float((u - exact).norm()) < tol                               # pass(3)
        u = sph_xyz(lambda x, y, z: jnp.cos(4 * z)).curl()
        exact = Spherefunv(sph_xyz(lambda x, y, z: -4 * y * jnp.sin(4 * z)),
                           sph_xyz(lambda x, y, z: 4 * x * jnp.sin(4 * z)),
                           sph_xyz(lambda x, y, z: 0 * x))
        assert float((u - exact).norm()) < tol                               # pass(4)
        u = sph_xyz(lambda x, y, z: jnp.cos(4 * x)).curl()
        exact = Spherefunv(sph_xyz(lambda x, y, z: 0 * x),
                           sph_xyz(lambda x, y, z: -4 * z * jnp.sin(4 * x)),
                           sph_xyz(lambda x, y, z: 4 * y * jnp.sin(4 * x)))
        assert float((u - exact).norm()) < 100 * tol                         # pass(5)
        u = sph_xyz(lambda x, y, z: jnp.cos(4 * y)).curl()
        exact = Spherefunv(sph_xyz(lambda x, y, z: 4 * z * jnp.sin(4 * y)),
                           sph_xyz(lambda x, y, z: 0 * x),
                           sph_xyz(lambda x, y, z: -4 * x * jnp.sin(4 * y)))
        assert float((u - exact).norm()) < 100 * tol                         # pass(6)
