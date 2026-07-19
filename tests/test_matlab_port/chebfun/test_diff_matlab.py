"""Port of MATLAB Chebfun tests/chebfun/test_diff.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunDiff:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert chebfun().diff().isempty()

    def test_piecewise_sin(self):
        f1 = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        df1 = f1.diff()
        err = jnp.abs(df1(XR) - jnp.cos(XR))
        assert float(jnp.max(err)) < 100 * df1.vscale * EPS

    def test_transpose_variant(self):
        # pass(3): diff of a row chebfun f1.' matches cos, staying a row.
        f1 = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        df1t = f1.T.diff()
        assert df1t.is_transposed
        err = jnp.abs(df1t(XR) - jnp.cos(XR))
        assert float(jnp.max(err)) < 100 * df1t.vscale * EPS

    def test_second_derivative(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(2 * x))
        d2 = f.diff(2)
        exact = jnp.exp(XR) * (4 * jnp.cos(2 * XR) - 3 * jnp.sin(2 * XR))
        err = jnp.abs(d2(XR) - exact)
        assert float(jnp.max(err)) < 1e4 * d2.vscale * EPS
