"""Port of MATLAB Chebfun tests/chebfun2/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS
D = [(-1.0, 1.0, -1.0, 1.0), (-2.0, 2.0, -2.0, 2.0),
     (-1.0, float(np.pi), 0.0, float(2 * np.pi))]


class TestChebfun2Times:
    def test_scalar_times(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        h = Chebfun2.from_function(lambda x, y: 2 * jnp.cos(x * y))
        assert float((f * 2 - h).norm()) < TOL
        assert float((2 * f - h).norm()) < TOL

    def test_square(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        k = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) ** 2)
        assert float((f ** 2 - k).norm()) < TOL
        assert float((f * f - k).norm()) < TOL

    @pytest.mark.parametrize("dom", D)
    def test_times_matches_direct_construction(self, dom):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=dom)
        g = Chebfun2.from_function(lambda x, y: x + y + x * y, domain=dom)
        ftg = Chebfun2.from_function(
            lambda x, y: jnp.cos(x * y) * (x + y + x * y), domain=dom)
        tolr = float(np.max(np.abs(dom))) * TOL
        assert float((f * g - ftg).norm()) < 10 * tolr
