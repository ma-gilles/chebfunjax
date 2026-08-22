"""Port of MATLAB Chebfun tests/ballfun/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.ballfun.ballfun import Ballfun

jax.config.update("jax_enable_x64", True)


class TestBallfunSize:
    def test_all_matlab_assertions(self):
        # Example 1
        f = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        assert f.size == (1, 1, 1)  # pass(1)

        # Example 2
        f = Ballfun.from_function(lambda x, y, z: x)
        assert f.size == (2, 3, 3)  # pass(2)
