"""Port of MATLAB Chebfun tests/spherefunv/test_empty.m (Fable 5).

FIXED: Spherefunv.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/spherefunv/test_empty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv


class TestSpherefunvEmpty:
    def test_empty_and_nonempty(self):
        assert Spherefunv.empty().isempty()
        v = Spherefunv(Spherefun.from_function(lambda lam, th: jnp.cos(th)), Spherefun.from_function(lambda lam, th: jnp.sin(th) * jnp.cos(lam)))
        assert not v.isempty()
