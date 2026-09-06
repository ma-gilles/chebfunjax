"""Port of MATLAB Chebfun tests/chebfun2/test_poldec.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_poldec.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestChebfun2Poldec:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = chebfun2(lambda x, y: jnp.exp(x * y))
        U, H = f.poldec()
        assert float(np.sum(np.abs(np.asarray(U.svd()) - 1))) < tol   # pass(1)
        assert float((U.mtimes(H) - f).norm()) < tol                  # pass(2)
