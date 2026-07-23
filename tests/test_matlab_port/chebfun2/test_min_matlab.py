"""Port of MATLAB Chebfun tests/chebfun2/test_min.m (Fable 5).

FIXED: Chebfun2.min2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Min:
    def test_min2(self):
        g = Chebfun2.from_function(
            lambda x, y: (x - 0.1) ** 2 + (y - 0.2) ** 2)
        v, loc = g.min2()
        assert abs(float(v)) < 1e-10
        np.testing.assert_allclose(np.asarray(loc), [0.1, 0.2],
                                   atol=1e-5)


class TestChebfun2MinDimensional:
    """MATLAB test_min.m: dimensional min returning 1D chebfuns."""

    TOL = float(np.sqrt(1000 * np.finfo(np.float64).eps))

    def test_cos_xy(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        t = np.linspace(-1.0, 1.0, 101)
        for h in (f.min(), f.min(None), f.min(None, 2)):
            assert float(np.max(np.abs(
                np.asarray(h(jnp.asarray(t))) - np.cos(t)))) < self.TOL
        assert f.min(None, 3) is f

    def test_linear_rect_domain(self):
        f = Chebfun2.from_function(lambda x, y: x + 0 * y,
                                   domain=(-2.0, 3.0, -4.0, 10.0))
        tx = np.linspace(-2.0, 3.0, 101)
        ty = np.linspace(-4.0, 10.0, 101)
        for h in (f.min(), f.min(None)):
            assert float(np.max(np.abs(
                np.asarray(h(jnp.asarray(tx))) - tx))) < self.TOL
        h3 = f.min(None, 2)
        assert float(np.max(np.abs(
            np.asarray(h3(jnp.asarray(ty))) + 2.0))) < self.TOL
        assert f.min(None, 3) is f
