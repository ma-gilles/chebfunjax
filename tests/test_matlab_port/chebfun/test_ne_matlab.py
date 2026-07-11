"""Port of MATLAB Chebfun tests/chebfun/test_ne.m (Fable 5).

FIXED: logical (indicator) chebfuns added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_ne.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.array([-0.5, 0.0, 0.5, 0.79, 0.9]))


class TestChebfunNe:
    def test_ne_indicator(self):
        f = cj.chebfun(jnp.sin)
        assert float(f.logical_ne(f)(jnp.asarray(0.3))) == 0.0
        g = cj.chebfun(jnp.cos)
        assert float(f.logical_ne(g)(jnp.asarray(0.3))) == 1.0
