"""Port of MATLAB Chebfun tests/chebfun3/test_isequal.m (Fable 5).

FIXED (Fable 5): Chebfun3.isequal added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Isequal:
    def test_all_matlab_assertions(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        g = Chebfun3.from_function(
            lambda x, y, z: jnp.sin(x + y ** 2 - z ** 3))

        assert f.isequal(f)
        assert not f.isequal(g)
