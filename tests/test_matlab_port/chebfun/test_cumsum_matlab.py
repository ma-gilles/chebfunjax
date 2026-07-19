"""Port of MATLAB Chebfun tests/chebfun/test_cumsum.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunCumsum:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert chebfun().cumsum().isempty()

    def test_piecewise_cos(self):
        f1 = cj.chebfun(jnp.cos, domain=[-1.0, -0.5, 0.5, 1.0])
        If1 = f1.cumsum()
        exact = jnp.sin(XR) - np.sin(-1.0)
        err = jnp.abs(If1(XR) - exact)
        assert float(jnp.max(err)) < 100 * If1.vscale * EPS

    def test_transpose_variant(self):
        # pass(3): cumsum of a row chebfun f1.' matches (sin - sin(-1)).'
        # -- and stays a row (isTransposed).
        f1 = cj.chebfun(jnp.cos, domain=[-1.0, -0.5, 0.5, 1.0])
        If1t = f1.T.cumsum()
        assert If1t.is_transposed
        exact = jnp.sin(XR) - np.sin(-1.0)
        err = jnp.abs(If1t(XR) - exact)
        assert float(jnp.max(err)) < 100 * If1t.vscale * EPS

    def test_array_valued(self):
        # pass(4): piecewise array-valued cumsum on [-1 -0.5 0.5 1].
        # FIXED (Fable 5, Big-Three array-valued epic): per-column offset chain.
        f3 = cj.chebfun(
            lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0.5, 1),
        )
        If3 = f3.cumsum()
        exact = jnp.stack(
            [jnp.sin(XR), jnp.cos(XR), jnp.exp(XR)], axis=-1
        ) - jnp.array([np.sin(-1.0), np.cos(-1.0), np.exp(-1.0)])
        err = jnp.abs(If3(XR) - exact)
        assert float(jnp.max(err)) < 10 * If3.vscale * EPS

    def test_cumsum_of_deriv_recovers(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        g = f.diff().cumsum()
        # g(x) = f(x) - f(-1)
        exact = f(XR) - float(f(jnp.asarray(-1.0)))
        err = jnp.abs(g(XR) - exact)
        assert float(jnp.max(err)) < 1e3 * f.vscale * EPS
