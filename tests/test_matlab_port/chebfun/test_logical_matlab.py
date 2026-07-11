"""Port of MATLAB Chebfun tests/chebfun/test_logical.m (Fable 5).

FIXED: logical (indicator) chebfuns added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_logical.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.array([-0.5, 0.0, 0.5, 0.79, 0.9]))


class TestChebfunLogical:
    def test_logical_of_nonzero(self):
        f = cj.chebfun(jnp.exp)
        ind = f.logical_ne(0.0)
        assert float(ind(jnp.asarray(0.4))) == 1.0
