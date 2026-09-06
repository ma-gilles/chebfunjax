"""Port of MATLAB Chebfun tests/spherefun/test_projection.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_projection.m
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


def _check(f, tol, scale):
    rng = np.random.RandomState(0)
    U = np.asarray(f.sample(16, 16)) + (1 - 2 * rng.rand(16, 16)) * 1e-10
    g = Spherefun.from_values(U)
    m = 101
    lam, th = np.meshgrid(np.linspace(-np.pi, np.pi, m), np.linspace(-np.pi, np.pi, m))
    V = np.asarray(g(jnp.asarray(lam), jnp.asarray(th)))
    assert np.std(V[-1, :]) <= tol                                           # pole
    assert np.std(V[(m - 1) // 2, :]) < tol                                  # pole
    id1 = slice(0, (m - 1) // 2 + 1)
    id2 = slice((m - 1) // 2, m)
    id3 = np.arange(m - 1, (m - 1) // 2 - 1, -1)
    assert np.max(np.abs(V[id1, id1] - V[id3][:, id2])) < scale * tol
    assert np.max(np.abs(V[id1, id2] - V[id3][:, id1])) < scale * tol


class TestSpherefunProjection:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        _check(sph_xyz(lambda x, y, z: 1 + 0 * x), tol, 1)                   # pass(1)-(3)
        _check(sph_xyz(lambda x, y, z: jnp.cos(np.pi * x * z) + jnp.sin(np.pi * x * y)), tol, 10)  # pass(4)-(6)
        f = sph_xyz(lambda x, y, z: jnp.cos(np.pi * x * z) + x * jnp.sin(np.pi * x * y))
        f = f.with_parity_indices(f.idx_minus, f.idx_plus)                   # swap idxPlus/idxMinus
        g = f.projectOntoBMCI()
        assert float(g.norm("inf")) < tol                                    # pass(7)
