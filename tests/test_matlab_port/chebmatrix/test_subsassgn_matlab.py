"""Port of MATLAB Chebfun tests/chebmatrix/test_subsassgn.m (Fable 5).

FIXED: ChebMatrix container API added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebmatrix/test_subsassgn.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebmatrixSubsassgn:
    def test_setitem(self):
        x = cj.chebfun(lambda t: t)
        ff = ChebMatrix.from_array([[x, x]])
        ff[0, 1] = x * 0
        assert abs(float(ff[0, 1](jnp.asarray(0.7)))) < 1e-15
        assert abs(float(ff[0, 0](jnp.asarray(0.7))) - 0.7) < 1e-14
