"""Port of MATLAB Chebfun tests/diskfun/test_BMCsvd.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_BMCsvd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref

from ._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunBMCsvd:
    def test_all_matlab_assertions(self):
        tol = 2e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        pi = np.pi
        f = disk_xy(lambda x, y: jnp.sin(5 * pi * x * y) + jnp.sin(pi * x ** 2 * jnp.cos(2 * (y - .1))))
        s = np.asarray(f.BMCsvd())
        assert s[-1] < 1e3 * tol                                             # pass(1)
        g = 100 * f
        t = np.asarray(g.BMCsvd())
        assert np.linalg.norm(s - t / 100) < tol                             # pass(2)
        u, s2, v = g.BMCsvd(return_uv=True)

        def ip(Q, i, j):
            return float((Q.cols[i - 1] * Q.cols[j - 1]).sum())
        total = (ip(u, 2, 27) + ip(u, 12, 7) + ip(v, 28, 21) + ip(v, 1, 2)
                 + ip(u, 19, 19) + ip(v, 1, 1) - 2)
        assert abs(total) < tol                                              # pass(3)
