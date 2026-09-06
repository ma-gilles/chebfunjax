"""Port of MATLAB Chebfun tests/spherefun/test_svd.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_svd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


class TestSpherefunSvd:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = sph_xyz(lambda x, y, z: jnp.cos(x * y * z))
        s = np.asarray(f.svd())
        assert abs(float(f.norm()) ** 2 - np.sum(s ** 2)) < tol             # pass(1)
        assert s[-1] < tol                                                   # pass(2)
        g = 100 * f
        t = np.asarray(g.svd())
        assert np.linalg.norm(s - t / 100) < tol                             # pass(3)
