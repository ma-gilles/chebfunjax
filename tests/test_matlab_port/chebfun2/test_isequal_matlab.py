"""Port of MATLAB Chebfun tests/chebfun2/test_isequal.m (Fable 5).

FIXED: Chebfun2.isequal added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Isequal:
    def test_isequal(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = Chebfun2.from_function(lambda x, y: jnp.sin(x + y ** 2))
        assert f.isequal(f) is True
        assert f.isequal(g) is False
