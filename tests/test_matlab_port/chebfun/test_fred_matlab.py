"""Port of MATLAB Chebfun tests/chebfun/test_fred.m (Fable 5).

FIXED: fred (Fredholm integral operator) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_fred.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunFred:
    def test_v4_reference_values(self):
        def K(u, v):
            return jnp.exp(-((u - v) ** 2))

        f = cj.chebfun(jnp.sin)
        F = cj.fred(K, f)
        vs = float(F.vscale)
        # pass(1)-(2): V4 reference values
        assert abs(float(F(jnp.asarray(0.5))) - 0.293968048825243) \
            < 1e1 * vs * np.finfo(float).eps
        assert abs(float(F.norm()) - 0.392002900508830) \
            < 1e1 * vs * np.finfo(float).eps
        # pass(3): onevar argument accepted
        cj.fred(K, f, 1)
