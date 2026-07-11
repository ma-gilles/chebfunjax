"""Port of MATLAB Chebfun tests/chebfun3/test_rank.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_rank.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Rank:
    def test_rank_one_separable(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.sin(z))
        assert f.rank == (1, 1, 1)

    def test_low_rank_bounded(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        rx, ry, rz = f.rank
        assert 1 <= rx and 1 <= ry and 1 <= rz
        m = max(int(c.n) for c in f.cols)
        assert max(rx, ry, rz) <= 2 * m
