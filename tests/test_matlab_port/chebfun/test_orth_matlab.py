"""Port of MATLAB Chebfun tests/chebfun/test_orth.m (Fable 5).

FIXED: Quasimatrix.orth added in the Fable 5 audit, with
complex-capable Householder QR/SVD.  The MATLAB test uses an
array-valued chebfun on a split domain [-1 0 1]; the chebfunjax
counterpart is a Quasimatrix on the single interval [-1, 1] (the
quasimatrix layer requires a single-interval domain), which carries
the same assertions.

Provenance
----------
MATLAB source : tests/chebfun/test_orth.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = np.finfo(float).eps
XS = jnp.asarray(np.linspace(-0.95, 0.95, 33))


def _build(cplx):
    s = cj.chebfun(jnp.sin)
    c = cj.chebfun(jnp.cos)
    third = cj.chebfun(lambda x: jnp.exp(1j * x)) if cplx \
        else cj.chebfun(jnp.exp)
    return Quasimatrix([s, c, third], s.domain)


class TestChebfunOrth:
    def _check(self, f, want_cols):
        Q = f.orth()
        assert Q.n_cols == want_cols
        G = np.array([
            [complex(np.asarray(Q.cols[i].inner(Q.cols[j])))
             for j in range(Q.n_cols)] for i in range(Q.n_cols)])
        # pass(1)/(3): Q'Q == I
        assert np.linalg.norm(G - np.eye(Q.n_cols)) < 10 * EPS * 10
        # pass(2)/(4): Q*(Q\\f) == f for every column of f
        for col in f.cols:
            coef = np.array([
                complex(np.asarray(Q.cols[i].inner(col)))
                for i in range(Q.n_cols)])
            rec = Q @ coef
            assert float(jnp.max(jnp.abs(
                rec.cols[0](XS) - col(XS)))) < 100 * EPS * 3

    def test_real_full_rank(self):
        self._check(_build(False), 3)

    def test_complex_rank_deficient(self):
        self._check(_build(True), 2)
