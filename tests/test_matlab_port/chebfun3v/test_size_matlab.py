"""Port of MATLAB Chebfun tests/chebfun3v/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

INF = float("inf")


class TestChebfun3vSize:
    def test_two_components(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(x + y + z))
        assert F.size() == (2, INF, INF, INF)

    def test_three_components(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.cos(z))
        assert F.size() == (3, INF, INF, INF)
