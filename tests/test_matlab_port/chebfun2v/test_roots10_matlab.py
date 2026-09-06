"""Port of MATLAB Chebfun tests/chebfun2v/test_roots10.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots10.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

pi = np.pi


class TestChebfun2vRoots10:
    def test_all_matlab_assertions(self):
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = chebfun2(lambda x, y: (x - 1) * (jnp.cos(x * y ** 2) + 2))
        g = chebfun2(lambda x, y: jnp.sin(8 * pi * y) * (jnp.cos(x * y) + 2))
        r2 = np.asarray(f.roots(g, method="resultant")).reshape(-1, 2)
        assert np.linalg.norm(np.sort(r2[:, 0]) - 1) < tol                   # pass(1)
        assert np.linalg.norm(np.sort(r2[:, 1]) - np.linspace(-1, 1, 17)) < tol  # pass(2)
