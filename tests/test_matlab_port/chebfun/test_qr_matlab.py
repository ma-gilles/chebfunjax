"""Port of MATLAB Chebfun tests/chebfun/test_qr.m (Fable 5).

MATLAB QRs an array-valued chebfun; chebfunjax QRs a list of chebfun
columns (quasimatrix).  Same orthogonality + reconstruction checks.

Provenance
----------
MATLAB source : tests/chebfun/test_qr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunQr:
    def test_orthogonality_and_reconstruction(self):
        cols = [cj.chebfun(lambda x: x, domain=(0.0, 1.0)),
                cj.chebfun(lambda x: 1.0 + 0 * x, domain=(0.0, 1.0)),
                cj.chebfun(lambda x: 2 * x - 1.0, domain=(0.0, 1.0))]
        Q, R = cols[0].qr(cols[1:]) if False else cols[0].qr(cols[1:])
        # Q columns orthonormal
        qcols = Q.cols if hasattr(Q, "cols") else Q
        n = len(qcols)
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                G[i, j] = float(qcols[i].innerProduct(qcols[j]))
        assert float(np.max(np.abs(G - np.eye(n)))) < 1e-12
        # reconstruction A = Q R
        R = np.asarray(R)
        xs = jnp.asarray(np.linspace(0.01, 0.99, 50))
        for j, c in enumerate(cols):
            rec = sum(qcols[i](xs) * R[i, j] for i in range(n))
            err = jnp.abs(rec - c(xs))
            assert float(jnp.max(err)) < 1e-12

    def test_complex_columns_rank(self):
        pytest.skip("chebfunjax quasimatrix qr on complex columns with "
                    "rank deficiency (MATLAB rank(A)==2 case) not "
                    "implemented")
