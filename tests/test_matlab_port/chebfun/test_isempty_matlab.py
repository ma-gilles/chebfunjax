"""Port of MATLAB Chebfun tests/chebfun/test_isempty.m (Fable 5).

FIXED: empty chebfun construction added in the Fable 5 audit --
chebfun(), chebfun([]), and n=0 all produce a piece-less Chebfun
with isempty() True.

Provenance
----------
MATLAB source : tests/chebfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj


class TestChebfunIsempty:
    def test_empty_constructions(self):
        assert cj.chebfun().isempty()
        assert cj.chebfun([]).isempty()
        assert cj.chebfun(jnp.sin, n=0).isempty()
        assert cj.chebfun([], domain=(-2, 2)).isempty()
        assert cj.chebfun([], domain=(3.14159, 42)).isempty()

    def test_nonempty(self):
        assert not cj.chebfun(jnp.sin).isempty()
        assert not cj.chebfun(jnp.sin, domain=(-2, 2)).isempty()
