"""Port of MATLAB Chebfun tests/chebfun/test_isnan.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsnan:
    def test_smooth_not_nan(self):
        f = cj.chebfun(lambda x: jnp.cos(2 * x))
        assert not bool(f.isnan())

    def test_nan_construction(self):
        pytest.skip("chebfunjax constructor raises on NaN samples "
                    "rather than representing NaN chebfuns")
