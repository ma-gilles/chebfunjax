"""Port of MATLAB Chebfun tests/ballfun/test_vscale.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_vscale.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.ballfun.ballfun import Ballfun

jax.config.update("jax_enable_x64", True)

TOL = 1e-2


class TestBallfunVscale:
    def test_all_matlab_assertions(self):
        # Example 1 : x
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(f.vscale - 1.0) < TOL  # pass(1)

        # Example 2 : -x^2
        f = Ballfun.from_function(lambda x, y, z: -x ** 2)
        assert abs(f.vscale - 1.0) < TOL  # pass(2)

        # Example 3 : 5*y
        f = Ballfun.from_function(lambda x, y, z: 5.0 * y)
        assert abs(f.vscale - 5.0) < TOL  # pass(3)
