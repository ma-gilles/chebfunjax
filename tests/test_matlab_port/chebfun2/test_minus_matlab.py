"""Port of MATLAB Chebfun tests/chebfun2/test_minus.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_minus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e5 * EPS
D = [(-1.0, 1.0, -1.0, 1.0), (-2.0, 2.0, -2.0, 2.0),
     (-1.0, float(np.pi), 0.0, float(2 * np.pi))]


class TestChebfun2Minus:
    @pytest.mark.parametrize("dom", D)
    def test_minus_matches_direct_construction(self, dom):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=dom)
        g = Chebfun2.from_function(lambda x, y: x + y + x * y, domain=dom)
        fmg = Chebfun2.from_function(
            lambda x, y: jnp.cos(x * y) - (x + y + x * y), domain=dom)
        tolr = float(np.max(np.abs(dom))) * TOL
        assert float((f - g - fmg).norm()) < 10 * tolr
