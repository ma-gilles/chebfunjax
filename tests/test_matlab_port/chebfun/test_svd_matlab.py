"""Port of MATLAB Chebfun tests/chebfun/test_svd.m (Fable 5).

Quasimatrix SVD (list of chebfun columns): U orthonormal, S descending,
reconstruction.

Provenance
----------
MATLAB source : tests/chebfun/test_svd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunSvd:
    def test_orthonormal_u_and_reconstruction(self):
        cols = [cj.chebfun(lambda x: x),
                cj.chebfun(lambda x: x ** 2),
                cj.chebfun(lambda x: jnp.exp(x))]
        out = cols[0].svd(cols[1:])
        U, S, V = out
        ucols = U.cols if hasattr(U, "cols") else U
        n = len(ucols)
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                G[i, j] = float(ucols[i].innerProduct(ucols[j]))
        assert float(np.max(np.abs(G - np.eye(n)))) < 1e-11
        S = np.asarray(S).ravel()
        assert bool(np.all(np.diff(S) <= 1e-12))
        V = np.asarray(V)
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 40))
        for j, c in enumerate(cols):
            rec = sum(ucols[i](xs) * S[i] * V[j, i] for i in range(n))
            err = jnp.abs(rec - c(xs))
            assert float(jnp.max(err)) < 1e-11
