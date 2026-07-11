"""Port of MATLAB Chebfun tests/chebfun/test_lt.m (Fable 5).

FIXED: logical (indicator) chebfuns added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_lt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.array([-0.5, 0.0, 0.5, 0.79, 0.9]))


class TestChebfunLt:
    def test_lt_indicator(self):
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        g = cj.chebfun(lambda x: 0 * x + np.sqrt(2) / 2)
        ind = f.lt(g)
        want = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(np.asarray(ind(XS)), want)

    def test_measure_identity(self):
        f = cj.chebfun(jnp.sin, domain=[-1.0, 0.0, 1.0])
        ind = f.gt(np.sqrt(2) / 2)
        assert abs(float(ind.sum()) - (1 - np.pi / 4)) < 1e-10
