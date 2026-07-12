"""Port of MATLAB Chebfun tests/diskfun/test_isempty.m (Fable 5).

FIXED: Diskfun.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/diskfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


class TestDiskfunIsempty:
    def test_empty_and_nonempty(self):
        assert Diskfun.empty().isempty()
        assert not Diskfun.from_function(lambda t, r: r * jnp.cos(t)).isempty()
