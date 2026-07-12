"""Port of MATLAB Chebfun tests/chebmatrix/test_cellfun.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_cellfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixCellfun:
    def test_cellfun(self):
        x = cj.chebfun(lambda t: t)
        ff = ChebMatrix.from_array([[x, x]])
        g = ff.cellfun(lambda b: b * 3)
        assert abs(float(g[0, 1](jnp.asarray(0.5))) - 1.5) < 1e-14
