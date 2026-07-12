"""Port of MATLAB Chebfun tests/chebfun/test_nextpow2.m (Fable 5).

FIXED: Chebfun.nextpow2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_nextpow2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

import chebfunjax as cj


class TestChebfunNextpow2:
    def test_piecewise_constant_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: 3.0 + 2 * x, domain=(0.0, 1.0))
            g = f.nextpow2()
        assert float(g(jnp.asarray(0.9))) == 3.0
        assert float(g(jnp.asarray(0.1))) == 2.0
