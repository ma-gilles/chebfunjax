"""Port of MATLAB Chebfun tests/chebfun2v/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

INF = float("inf")


class TestChebfun2vSize:
    def test_two_components(self):
        # pass(1)-(4): size(F) == [2 inf inf].
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y))
        assert F.shape == (2, INF, INF)
        assert F.shape[0] == 2

    def test_three_components(self):
        # pass(5)-(8): size(F) == [3 inf inf].
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y),
                                     lambda x, y: jnp.cos(x))
        assert F.shape == (3, INF, INF)
        assert F.shape[0] == 3
