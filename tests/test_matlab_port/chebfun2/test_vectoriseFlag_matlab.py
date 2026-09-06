"""Port of MATLAB Chebfun tests/chebfun2/test_vectoriseFlag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_vectoriseFlag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestChebfun2VectoriseFlag:
    def test_all_matlab_assertions(self):
        tol = 10 * ChebfunPref().cheb2Prefs.chebfun2eps
        f1 = chebfun2(lambda x, y: x)
        f2 = chebfun2(lambda x, y: x, vectorize=True)
        f3 = chebfun2(lambda x, y: x, domain=(-1, 1, -1, 1), vectorize=True)
        f4 = chebfun2(lambda x, y: x, vectorize=True, domain=(-1, 1, -1, 1))
        assert float((f1 - f2).norm()) < tol                          # pass(1)
        assert float((f1 - f3).norm()) < tol                          # pass(2)
        assert float((f1 - f4).norm()) < tol                          # pass(3)

        f1 = chebfun2(lambda x, y: x * y)
        f2 = chebfun2(lambda x, y: x * y, vectorize=True)
        assert float((f1 - f2).norm()) < tol                          # pass(4)

        g = chebfun2(lambda z: jnp.sum(z ** jnp.arange(10)), domain=tuple(np.pi / 2 * np.array([-1, 1, -1, 1])),
                     vectorize=True)
        r = np.asarray(g.roots())
        assert np.all(np.abs(np.abs(r) - 1) < 50 * tol)                # pass(5)
        assert abs(np.sum(r) + 1) < 100 * tol                          # pass(6)
