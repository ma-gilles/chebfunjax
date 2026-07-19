"""Port of MATLAB Chebfun tests/chebfun/test_isequal.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj


class TestChebfunIsequal:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun()
        assert f.isequal(f)

    def test_self_equality(self):
        f = cj.chebfun(lambda x: jnp.sin(x) * (x - 0.1))
        assert bool(f.isequal(f))

    def test_transpose_inequality(self):
        # pass(3): ~isequal(f, f.') -- a column and its row transpose differ.
        f = cj.chebfun(lambda x: jnp.sin(x) * (x - 0.1))
        assert not bool(f.isequal(f.T))
        assert bool(f.T.isequal(f.T))

    def test_different_functions(self):
        f = cj.chebfun(lambda x: jnp.sin(x) * (x - 0.1))
        g = cj.chebfun(jnp.sin)
        assert not bool(f.isequal(g))
