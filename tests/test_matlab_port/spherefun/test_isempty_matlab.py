"""Port of MATLAB Chebfun tests/spherefun/test_isempty.m (Fable 5).

FIXED: Spherefun.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/spherefun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunIsempty:
    def test_empty_and_nonempty(self):
        assert Spherefun.empty().isempty()
        assert not Spherefun.from_function(lambda lam, th: jnp.cos(th)).isempty()
