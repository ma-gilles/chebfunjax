"""Port of MATLAB Chebfun tests/chebfun/test_eq.m (Fable 5).

MATLAB f == g returns a logical chebfun (pointwise equality regions);
chebfunjax has no pointwise-eq chebfun -- roots of f - g cover the
underlying assertion.

Provenance
----------
MATLAB source : tests/chebfun/test_eq.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunEq:
    def test_pointwise_eq_chebfun(self):
        pytest.skip("chebfunjax has no pointwise == returning a logical "
                    "chebfun")

    def test_equality_locations_via_roots(self):
        # MATLAB: sin(x) == sqrt(2)/2 has solutions where sin crosses
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        c = float(np.sqrt(2) / 2)
        r = np.asarray((f - c).roots())
        exact = np.array([np.pi / 4])
        assert len(r) == 1
        assert abs(r[0] - exact[0]) < 100 * EPS
