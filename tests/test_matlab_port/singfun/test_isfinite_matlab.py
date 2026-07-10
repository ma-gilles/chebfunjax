"""Port of MATLAB Chebfun tests/singfun/test_isfinite.m (Opus 4.8).

chebfunjax Singfun has no ``isfinite`` method, but the MATLAB algorithm is
``out = ~any(exponents < 0) && isfinite(smoothPart)``.  Since the smooth part
is always finite, this reduces to "no negative exponent", which we compute
directly from ``f.exponents``.

Provenance
----------
MATLAB source : tests/singfun/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun

A = 0.64
D = -1.28


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _isfinite(f):
    # MATLAB @singfun/isfinite: ~any(exponents < 0) && isfinite(smoothPart)
    return all(e >= 0 for e in f.exponents)


class TestSingfunIsfinite:
    def test_frac_root_left_is_finite(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        assert _isfinite(f)

    def test_frac_pole_left_not_finite(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        assert not _isfinite(f)
