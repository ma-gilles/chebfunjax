"""Port of MATLAB Chebfun tests/chebfun/test_iszero.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIszero:
    def test_zero_constant(self):
        f = cj.chebfun(lambda x: jnp.zeros_like(x))
        assert bool(f.iszero())

    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_nonzero_constant(self):
        f = cj.chebfun(lambda x: 2.0 + 0 * x)
        assert not bool(f.iszero())

    def test_piecewise_zero(self):
        f = cj.chebfun(lambda x: jnp.zeros_like(x),
                       domain=[-1.0, 0.0, 1.0])
        assert bool(f.iszero())
