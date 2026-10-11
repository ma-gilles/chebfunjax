"""Port of MATLAB Chebfun tests/ballfun/test_isequal.m (Fable 5).

FIXED (Fable 5): Ballfun.isequal added in the audit.

Provenance
----------
MATLAB source : tests/ballfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunIsequal:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_values(jnp.ones((21, 20, 22)))
        g = f + f - f
        assert f.isequal(g) and g.isequal(f)
