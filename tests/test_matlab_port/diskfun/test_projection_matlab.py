"""Port of MATLAB Chebfun tests/diskfun/test_projection.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_projection.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.utils.quadrature import chebpts

from ._cart import disk_xy

jax.config.update("jax_enable_x64", True)


def _check(f, tol, scale):
    rng = np.random.RandomState(0)
    U = np.asarray(f.sample(16, 17)) + (1 - 2 * rng.rand(17, 16)) * 1e-10
    g = Diskfun.from_values(U)
    m = 101
    t, r = np.meshgrid(np.linspace(-np.pi, np.pi, m), np.asarray(chebpts(m)))
    V = np.asarray(g(jnp.asarray(t), jnp.asarray(r)))
    assert np.std(V[(m - 1) // 2, :]) < tol
    id1 = slice(0, (m - 1) // 2 + 1)
    id2 = slice((m - 1) // 2, m)
    id3 = np.arange(m - 1, (m - 1) // 2 - 1, -1)
    assert np.max(np.abs(V[id1, id1] - V[id3][:, id2])) < scale * tol
    assert np.max(np.abs(V[id1, id2] - V[id3][:, id1])) < scale * tol


class TestDiskfunProjection:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        _check(disk_xy(lambda x, y: 1 + 0 * x), tol, 1)                      # pass(1)-(2)
        _check(disk_xy(lambda x, y: jnp.cos(np.pi * x * y) + jnp.sin(3 * np.pi * x)), tol, 10)  # pass(3)-(4)
        f = disk_xy(lambda x, y: jnp.cos(np.pi * x * y) + x * jnp.sin(np.pi * x * y))
        f = f.with_parity_indices(f.idx_minus, f.idx_plus)
        g = f.projectOntoBMCII()
        assert float(g.norm("inf")) < tol                                    # pass(5)
