"""Port of MATLAB Chebfun tests/chebfun/test_not.m (Fable 5).

FIXED: logical (indicator) chebfuns added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_not.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.array([-0.5, 0.0, 0.5, 0.79, 0.9]))


class TestChebfunNot:
    def test_not_indicator(self):
        z = cj.chebfun(lambda x: jnp.zeros_like(x))
        f = cj.chebfun(jnp.exp)
        assert float(z.logical_not()(jnp.asarray(0.3))) == 1.0
        assert float(f.logical_not()(jnp.asarray(0.3))) == 0.0
