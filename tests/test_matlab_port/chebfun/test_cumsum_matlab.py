"""Port of MATLAB Chebfun tests/chebfun/test_cumsum.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunCumsum:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_piecewise_cos(self):
        f1 = cj.chebfun(jnp.cos, domain=[-1.0, -0.5, 0.5, 1.0])
        If1 = f1.cumsum()
        exact = jnp.sin(XR) - np.sin(-1.0)
        err = jnp.abs(If1(XR) - exact)
        assert float(jnp.max(err)) < 100 * If1.vscale * EPS

    def test_transpose_variant(self):
        pytest.skip("chebfunjax has no row-chebfun transpose")

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued Chebfun")

    def test_cumsum_of_deriv_recovers(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        g = f.diff().cumsum()
        # g(x) = f(x) - f(-1)
        exact = f(XR) - float(f(jnp.asarray(-1.0)))
        err = jnp.abs(g(XR) - exact)
        assert float(jnp.max(err)) < 1e3 * f.vscale * EPS
