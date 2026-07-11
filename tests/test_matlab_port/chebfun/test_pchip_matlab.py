"""Port of MATLAB Chebfun tests/chebfun/test_pchip.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_pchip.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunPchip:
    def test_interpolates_data(self):
        x = jnp.arange(11.0)
        y = jnp.sin(x)
        f = cj.chebfun(lambda t: t, domain=(0.0, 10.0)).pchip(x, y)
        err = jnp.abs(f(x) - y)
        assert float(jnp.max(err)) < 100 * EPS
        assert len(f.funs) == 10

    def test_monotone_data_stays_monotone(self):
        # pchip's defining property (shape preservation)
        x = jnp.arange(6.0)
        y = jnp.asarray([0.0, 1.0, 1.2, 1.21, 3.0, 5.0])
        f = cj.chebfun(lambda t: t, domain=(0.0, 5.0)).pchip(x, y)
        xs = jnp.asarray(np.linspace(0.01, 4.99, 200))
        d = f.diff()
        assert float(jnp.min(d(xs))) > -1e-10

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued chebfun")
