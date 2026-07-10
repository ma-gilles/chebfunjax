"""Port of MATLAB Chebfun tests/singfun/test_isnan.m (Opus 4.8).

chebfunjax Singfun has no ``isnan`` method, but the MATLAB algorithm is
``out = isnan(smoothPart) || any(isnan(exponents))``, which we compute
directly from the smooth-part coefficients and the exponents.  The third case
genuinely exercises chebfunjax arithmetic: ``NaN * f`` propagates NaN into the
smooth-part coefficients.

Provenance
----------
MATLAB source : tests/singfun/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun

A = 0.64
D = -1.28


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _isnan(f):
    # MATLAB @singfun/isnan: isnan(smoothPart) || any(isnan(exponents))
    return bool(jnp.any(jnp.isnan(f.coeffs))) or any(math.isnan(e) for e in f.exponents)


class TestSingfunIsnan:
    def test_frac_root_left_not_nan(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        assert not _isnan(f)

    def test_frac_pole_left_not_nan(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        assert not _isnan(f)

    def test_nan_times_singfun_is_nan(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        g = float("nan") * f
        assert _isnan(g)
