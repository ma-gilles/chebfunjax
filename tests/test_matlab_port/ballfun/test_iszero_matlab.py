"""Port of MATLAB Chebfun tests/ballfun/test_iszero.m (Fable 5).

FIXED (Fable 5): Ballfun.iszero added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunIszero:
    def test_all_matlab_assertions(self):
        # f - f is exactly zero.
        f = Ballfun.from_values(jnp.ones((21, 20, 22)))
        assert (f - f).iszero()

        # 1e-20 is a (tiny) nonzero constant.
        assert not Ballfun.from_values(1e-20).iszero()

        # The zero function.
        assert Ballfun.from_values(0.0).iszero()
