"""Port of MATLAB Chebfun tests/spherefun/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


class TestSpherefunNorm:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = sph_xyz(lambda x, y, z: 1 + 0 * x)
        assert abs(np.sqrt(float(f.sum2())) - float(f.norm())) < tol         # pass(1)
        f = sph_xyz(lambda x, y, z: jnp.cos(x * y * z))
        s = np.asarray(f.svd())
        assert abs(np.sum(s ** 2) - float(f.norm()) ** 2) < tol              # pass(2)
        f = sph_xyz(lambda x, y, z: x + y + z)
        assert abs(float(f.norm("inf")) - np.sqrt(3)) < tol                  # pass(3)

    def test_norm_of_cos_theta(self):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        assert abs(float(f.norm()) - np.sqrt(4 * np.pi / 3)) < 1e-12
