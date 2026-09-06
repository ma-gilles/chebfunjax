"""Port of MATLAB Chebfun tests/spherefun/test_BMCsvd.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_BMCsvd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


class TestSpherefunBMCsvd:
    def test_all_matlab_assertions(self):
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = sph_xyz(lambda x, y, z: jnp.sin(np.pi * x * y) + jnp.sin(np.pi * x * z))
        s = np.asarray(f.BMCsvd())
        assert s[-1] < 1e3 * tol                                             # pass(1)
        g = 100 * f
        t = np.asarray(g.BMCsvd())
        assert np.linalg.norm(s - t / 100) < tol                             # pass(2)
