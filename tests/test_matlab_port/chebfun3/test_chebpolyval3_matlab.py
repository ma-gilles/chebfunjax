"""Port of MATLAB Chebfun tests/chebfun3/test_chebpolyval3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_chebpolyval3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestChebfun3Chebpolyval3:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
        rng = np.random.RandomState(0)
        exact = rng.rand(4, 4, 4)
        vals = np.asarray(chebfun3(jnp.asarray(exact)).chebpolyval3())
        assert np.linalg.norm(exact.ravel() - vals.ravel()) < 10 * tol      # pass(1)
        A = rng.rand(3, 4, 5)
        B = np.asarray(chebfun3(jnp.asarray(A)).chebpolyval3())
        assert np.linalg.norm(A.ravel() - B.ravel()) < tol                  # pass(2)
