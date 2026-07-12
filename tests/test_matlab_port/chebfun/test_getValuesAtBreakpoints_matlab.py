"""Port of MATLAB Chebfun tests/chebfun/test_getValuesAtBreakpoints.m
(Fable 5).

FIXED: getValuesAtBreakpoints added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_getValuesAtBreakpoints.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunGetValuesAtBreakpoints:
    def test_identity(self):
        # pass(1)
        f = cj.chebfun(lambda x: x, domain=[-1.0, 0.0, 0.5, 1.0])
        v = np.asarray(cj.getValuesAtBreakpoints(f))
        np.testing.assert_allclose(v, [-1.0, 0.0, 0.5, 1.0],
                                   atol=1e-15)

    def test_custom_op(self):
        # pass(3)-(4)
        f = cj.chebfun(lambda x: x, domain=[-1.0, 0.0, 0.5, 1.0])

        def op(x):
            return x + 100.0 * (x == 0)

        v = np.asarray(cj.getValuesAtBreakpoints(f, op))
        np.testing.assert_allclose(v, [-1.0, 100.0, 0.5, 1.0],
                                   atol=1e-15)
        v2 = np.asarray(cj.getValuesAtBreakpoints(f, jnp.sign))
        np.testing.assert_allclose(v2, [-1.0, 0.0, 1.0, 1.0],
                                   atol=1e-15)
