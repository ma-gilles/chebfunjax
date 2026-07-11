"""Port of MATLAB Chebfun tests/chebfun/test_isempty.m (Fable 5).

chebfunjax has no empty chebfun; isempty() exists and must return
False for any constructed chebfun.

Provenance
----------
MATLAB source : tests/chebfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsempty:
    def test_empty_constructions(self):
        pytest.skip("chebfunjax has no empty chebfun constructions")

    def test_nonempty_is_false(self):
        f = cj.chebfun(jnp.sin)
        assert not bool(f.isempty())
