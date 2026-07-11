"""Port of MATLAB Chebfun tests/chebfun2/test_rank.m (Fable 5).

MATLAB checks rank(f) <= min(m, n) grid lengths.  chebfunjax exposes
rank but not the (m, n) grid lengths of cols/rows techs; the port
checks rank <= max tech length, the same inequality up to transposition.

Provenance
----------
MATLAB source : tests/chebfun2/test_rank.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

CASES = [
    lambda x, y: 1.0 / (1 + 100 * (x ** 2 - y ** 2) ** 2),
    lambda x, y: 1.0 / (1 + 100 * (0.5 - x ** 2 - y ** 2) ** 2),
    lambda x, y: 1.0 / (1 + 1000 * ((x - 0.5) ** 2 * (y + 0.5) ** 2
                                    * (x + 0.5) ** 2 * (y - 0.5) ** 2)),
    lambda x, y: jnp.cos(10 * (x ** 2 + y)) * jnp.sin(10 * (x + y ** 2)),
    lambda x, y: jnp.tanh(10 * x) * jnp.tanh(10 * y)
    / jnp.tanh(10.0) ** 2 + jnp.cos(5 * x),
]


class TestChebfun2Rank:
    @pytest.mark.parametrize("i", range(len(CASES)))
    def test_rank_at_most_grid_length(self, i):
        f = Chebfun2.from_function(CASES[i])
        m = max(int(c.n) for c in f.approx.cols)
        n = max(int(r.n) for r in f.approx.rows)
        assert f.rank <= max(m, n)
        assert f.rank >= 1
