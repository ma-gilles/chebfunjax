"""Port of MATLAB Chebfun tests/diskfun/test_biharm.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_biharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.utils.quadrature import chebpts

jax.config.update("jax_enable_x64", True)


def _points(m, n):
    x = -np.pi + 2 * np.pi * np.arange(2 * n) / (2 * n)
    y = np.asarray(chebpts(m))
    return x, y[int(np.ceil(m / 2)) - 1:]


def _sample_error(h, g):
    m = 7
    n = m - 1
    x, y = _points(m, n)
    L2, T2 = np.meshgrid(x, y)
    F = np.asarray(h(jnp.asarray(L2), jnp.asarray(T2)))
    approx = np.asarray(g.fevalm(jnp.asarray(x), jnp.asarray(y)))
    return np.max(np.abs(F.ravel() - approx.ravel()))


class TestDiskfunBiharm:
    def test_all_matlab_assertions(self):
        tol = 1e7 * ChebfunPref().cheb2Prefs.chebfun2eps
        for ell in (1, 2, 3):
            for m in range(1, ell + 1):
                a = np.sqrt((3 / 4) ** 2 * np.pi ** 2 + ell ** 2)
                b = (m + ell / 2) * np.pi
                jz = np.asarray(chebfun(lambda x, _l=ell: jnp.asarray(jv(_l, np.asarray(x))),
                                        domain=(a, b)).roots())
                jzero = float(jz[m - 1])
                f = Diskfun.harmonic(ell, m)
                lap2 = f.biharm()
                assert _sample_error(jzero ** 4 * f, lap2) < 2 * jzero ** 4 * tol
