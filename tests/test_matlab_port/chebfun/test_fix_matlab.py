"""Port of MATLAB Chebfun tests/chebfun/test_fix.m (Fable 5).

FIXED: Chebfun.fix added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_fix.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunFix:
    def test_fix_rounds_toward_zero(self):
        f = cj.chebfun(lambda x: 2.5 * x)
        xs = jnp.asarray(np.array([-0.9, -0.3, 0.3, 0.9]))
        np.testing.assert_allclose(np.asarray(f.fix()(xs)),
                                   [-2.0, 0.0, 0.0, 2.0])
