"""Port of MATLAB Chebfun tests/singfun/test_isequal.m (Opus 4.8).

All five source predicates use the public isequal method.

Provenance
----------
MATLAB source : tests/singfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunIsequal:
    def test_empty_equal(self):
        assert Singfun.empty().isequal(Singfun.empty())

    def test_zerosingfun_equal(self):
        assert Singfun.zeroSingFun().isequal(Singfun.zeroSingFun())

    def test_identical_nonzero_equal(self):
        f = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        g = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        assert f.isequal(g)

    def test_different_exponents_not_equal(self):
        f = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        g = _sf(lambda x: 1.0 / (1 + x), (-1.8, 0.0))
        assert not f.isequal(g)

    def test_different_smoothpart_not_equal(self):
        f = _sf(lambda x: jnp.cos(x) / (1 + x), (-1.0, 0.0))
        g = _sf(lambda x: jnp.sin(x) / (1 + x), (-1.0, 0.0))
        assert not f.isequal(g)
