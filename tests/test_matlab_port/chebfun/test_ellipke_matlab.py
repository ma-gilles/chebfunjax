"""Port of MATLAB Chebfun tests/chebfun/test_ellipke.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_ellipke.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import ellipk

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunEllipke:
    def test_ellipke_of_identity(self):
        m = cj.chebfun(lambda x: x, domain=(0.0, 0.99))
        K1 = m.ellipke()
        K1 = K1[0] if isinstance(K1, tuple) else K1
        xs = jnp.asarray(np.linspace(0.01, 0.98, 60))
        exact = jnp.asarray(ellipk(np.asarray(xs)))
        err = jnp.abs(K1(xs) - exact)
        assert float(jnp.max(err)) < 1e2 * EPS * K1.vscale

    def test_array_valued(self):
        # pass(2): array-valued ellipke composition on a piecewise domain
        # (-1:.5:1) of .05+abs(.9*[sin(pi x) cos(pi x)]).
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) piecewise chebfun.
        dom = tuple(np.arange(-1.0, 1.01, 0.5))
        f = cj.chebfun(
            lambda x: 0.05
            + jnp.abs(0.9 * jnp.stack([jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1)),
            domain=dom,
        )
        K1 = f.ellipke()
        K1 = K1[0] if isinstance(K1, tuple) else K1
        xs = jnp.asarray(np.linspace(-0.97, 0.97, 80))
        exact = ellipk(
            0.05
            + np.abs(0.9 * np.stack([np.sin(np.pi * np.asarray(xs)), np.cos(np.pi * np.asarray(xs))], axis=-1))
        )
        err = float(jnp.max(jnp.abs(K1(xs) - jnp.asarray(exact))))
        assert err < 1e3 * EPS * K1.vscale
