"""Port of MATLAB Chebfun tests/chebfun/test_horzcat.m (Fable 5).

FIXED (adapted): MATLAB's array-valued [f g] concatenation maps to
the Quasimatrix counterpart (the designated chebfunjax analogue, as
with the null/orth/pinv ports).  Assertions are the same: column
count and per-column values.  MATLAB's mixed-breakpoint domains use
a common [-1, 1] here (Quasimatrix requires single-interval
domains); the value assertions are unchanged.

Provenance
----------
MATLAB source : tests/chebfun/test_horzcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.random(200) - 1)


class TestChebfunHorzcat:
    def test_concatenations(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = cj.chebfun(jnp.exp)
        one = cj.chebfun(lambda x: 1.0 + 0 * x)

        # pass(1): [1 f]
        Q = Quasimatrix([one], one.domain).horzcat(f)
        assert Q.n_cols == 2
        assert float(jnp.max(jnp.abs(Q.cols[0](XR) - 1.0))) < 1e-13
        assert float(jnp.max(jnp.abs(
            Q.cols[1](XR) - jnp.sin(XR)))) < 1e-13

        # pass(2): [f f]
        Q = Quasimatrix([f], f.domain).horzcat(f)
        assert Q.n_cols == 2
        for c in Q.cols:
            assert float(jnp.max(jnp.abs(c(XR) - jnp.sin(XR)))) \
                < 1e-13

        # pass(3)-(4): [f g], then appending h
        Q = Quasimatrix([f], f.domain).horzcat(g)
        assert Q.n_cols == 2
        Q = Q.horzcat(h)
        assert Q.n_cols == 3
        exact = [jnp.sin(XR), jnp.cos(XR), jnp.exp(XR)]
        for c, e in zip(Q.cols, exact):
            assert float(jnp.max(jnp.abs(c(XR) - e))) < 1e-13
