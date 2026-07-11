"""Port of MATLAB Chebfun tests/chebfun/test_le.m (Fable 5).

FIXED: logical (indicator) chebfuns added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_le.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.array([-0.5, 0.0, 0.5, 0.79, 0.9]))


class TestChebfunLe:
    def test_le_indicator(self):
        f = cj.chebfun(jnp.sin, domain=[-1.0, 0.0, 1.0])
        ind = f.le(0.0)
        np.testing.assert_allclose(
            np.asarray(ind(jnp.asarray(np.array([-0.5, 0.5])))),
            [1.0, 0.0])
