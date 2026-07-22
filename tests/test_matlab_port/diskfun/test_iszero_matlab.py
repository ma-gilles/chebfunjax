"""Port of MATLAB Chebfun tests/diskfun/test_iszero.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunIszero:
    def test_zero_is_zero(self):
        # pass(1): f = 0 -> iszero(f)
        f = _df(lambda t, r: 0.0 * r)
        assert f.iszero()

    def test_cos_x_not_zero(self):
        # pass(2): f = cos(x) -> ~iszero(f)
        f = _df(lambda t, r: jnp.cos(r * jnp.cos(t)))
        assert not f.iszero()
