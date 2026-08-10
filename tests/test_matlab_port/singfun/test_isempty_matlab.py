"""Port of MATLAB Chebfun tests/singfun/test_isempty.m (Opus 4.8).

MATLAB distinguishes an empty singfun (``singfun()``), a zero singfun
(``singfun.zeroSingFun()``), and a non-empty singfun.

Provenance
----------
MATLAB source : tests/singfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun


class TestSingfunIsempty:
    def test_empty_is_empty(self):
        f = Singfun.empty()
        assert f.isempty()

    def test_zerosingfun_not_empty(self):
        f = Singfun.zeroSingFun()
        assert not f.isempty()

    def test_nonzero_not_empty(self):
        f = Singfun.from_function(
            lambda x: 1.0 / (1 + x), exponents=(-1.0, 0.0)
        )
        assert not f.isempty()
        assert not f.iszero()
        assert bool(jnp.isfinite(f.smoothPart(jnp.float64(0.0))))
