"""Port of MATLAB Chebfun tests/chebfun/test_volt.m (Fable 5).

FIXED: volt (Volterra integral operator) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_volt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunVolt:
    def test_v4_reference_values(self):
        def K(u, v):
            return jnp.exp(-((u - v) ** 2))

        f = cj.chebfun(jnp.sin)
        F = cj.volt(K, f)
        vs = max(float(F.vscale), 0.3)
        # pass(1)-(2): V4 reference values
        assert abs(float(F(jnp.asarray(0.5))) - (-0.013808570536509)) \
            < 10 * vs * np.finfo(float).eps
        assert abs(float(F.norm()) - 0.334612395278957) \
            < 10 * vs * np.finfo(float).eps
        # pass(3): onevar argument accepted
        cj.volt(K, f, 1)
