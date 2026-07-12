"""Port of MATLAB Chebfun tests/diskfunv/test_empty.m (Fable 5).

FIXED: Diskfunv.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/diskfunv/test_empty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv


class TestDiskfunvEmpty:
    def test_empty_and_nonempty(self):
        assert Diskfunv.empty().isempty()
        v = Diskfunv(Diskfun.from_function(lambda t, r: r * jnp.cos(t)), Diskfun.from_function(lambda t, r: r ** 2))
        assert not v.isempty()
