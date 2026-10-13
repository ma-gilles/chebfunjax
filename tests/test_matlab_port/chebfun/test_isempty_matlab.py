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

import math

import jax.numpy as jnp

import chebfunjax as cj


class TestChebfunIsempty:
    def test_empty_constructions(self):
        assert cj.chebfun().isempty()
        assert cj.chebfun([]).isempty()
        assert cj.chebfun(jnp.sin, n=0).isempty()
        assert cj.chebfun([], domain=(-2, 2)).isempty()
        assert cj.chebfun([], domain=(math.pi, 42)).isempty()

    def test_nonempty(self):
        assert not cj.chebfun(jnp.sin).isempty()
        assert not cj.chebfun(jnp.sin, domain=(-2, 2)).isempty()

    def test_fixed_length_one_is_nonempty(self):
        # Native test_isempty pass(8): chebfun(@sin, [-2 2], 1).
        assert not cj.chebfun(jnp.sin, domain=(-2, 2), n=1).isempty()

    def test_array_valued_chebfun_is_nonempty(self):
        # Native pass(9): [x abs(x)] is a two-column, nonempty Chebfun.
        x = cj.chebfun("x")
        f = cj.Chebfun.horzcat(x, abs(x))
        assert not f.isempty()

    def test_singular_finite_domain_is_nonempty(self):
        # Native pass(10); its exponent -1.64 differs from feval pass(31).
        dom = (-2.0, 7.0)
        power = -1.64
        f = cj.chebfun(
            lambda x: (x - dom[0]) ** power,
            domain=dom,
            exps=(power, 0),
            splitting=True,
        )
        assert not f.isempty()
