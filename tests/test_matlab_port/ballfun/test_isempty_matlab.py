"""Port of MATLAB Chebfun tests/ballfun/test_isempty.m (Fable 5).

Original slots cover empty factory and empty numeric input.
The existing nonempty x construction remains an independent regression.

Provenance
----------
MATLAB source : tests/ballfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunIsempty:
    def test_empty_and_nonempty(self):
        assert Ballfun.empty().isempty()
        f = Ballfun.from_function(lambda x, y, z: x)
        assert not f.isempty()

    def test_empty_numeric_coefficients(self):
        # Native ballfun([]) returns before inspecting constructor flags.
        assert Ballfun.from_coeffs(jnp.empty((0, 0, 0))).isempty()
