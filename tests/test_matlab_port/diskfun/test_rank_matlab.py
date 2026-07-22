"""Port of MATLAB Chebfun tests/diskfun/test_rank.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_rank.m
Chebfun commit: 7574c77

MATLAB uses ``k = length(f)`` for the rank and ``[m, n] = length(f)`` for
the row/column slice lengths; chebfunjax exposes these as ``f.rank`` and
``f.length()`` respectively.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


def _x(t, r):
    return r * jnp.cos(t)


def _y(t, r):
    return r * jnp.sin(t)


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunRank:
    def _check(self, f):
        k = f.rank
        m, n = f.length()
        assert k <= min(m, n)

    def test_rank_bound_1(self):
        c = 10.0
        self._check(_df(lambda t, r: 1.0 / (1 + c * ((_x(t, r) - 1) ** 2 - _y(t, r) ** 2) ** 2)))

    def test_rank_bound_2(self):
        c = 100.0
        self._check(_df(lambda t, r: 1.0 / (1 + c * (_x(t, r) ** 2 - _y(t, r) ** 2) ** 2)))

    def test_rank_bound_3(self):
        self._check(_df(lambda t, r: jnp.cos(10 * _x(t, r) ** 2) * jnp.sin(10 * (_x(t, r) + _y(t, r) ** 2))))

    def test_rank_bound_4(self):
        self._check(
            _df(lambda t, r: jnp.tanh(20 * _x(t, r)) * jnp.tanh(10 * _y(t, r)) * jnp.cos(50 * _x(t, r) * _y(t, r) + 1))
        )
