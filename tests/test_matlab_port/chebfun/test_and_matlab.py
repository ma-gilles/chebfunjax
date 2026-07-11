"""Port of MATLAB Chebfun tests/chebfun/test_and.m (Fable 5).

FIXED: logical (indicator) chebfuns added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_and.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.array([-0.5, 0.0, 0.5, 0.79, 0.9]))


class TestChebfunAnd:
    def test_and_indicator(self):
        f = cj.chebfun(jnp.exp)
        z = cj.chebfun(lambda x: jnp.zeros_like(x))
        assert float(f.logical_and(f)(jnp.asarray(0.2))) == 1.0
        assert float(f.logical_and(z)(jnp.asarray(0.2))) == 0.0
