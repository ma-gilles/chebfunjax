"""Port of MATLAB Chebfun tests/chebfun2/test_chebpts2.m (Fable 5).

MATLAB chebfun2.chebpts2(n[, m, dom]) returns meshgrid tensors of
Chebyshev points.  chebfunjax exposes 1-D chebpts; the port builds the
meshgrid the same way and checks the same identities.

Provenance
----------
MATLAB source : tests/chebfun2/test_chebpts2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS


class TestChebfun2Chebpts2:
    def test_square_grid(self):
        n = 10
        x = jnp.asarray(chebpts(n, kind=2))
        xx1, yy1 = jnp.meshgrid(x, x)
        xx2, yy2 = jnp.meshgrid(x, x)
        assert float(jnp.max(jnp.abs(xx1 - xx2))) < TOL
        assert float(jnp.max(jnp.abs(yy1 - yy2))) < TOL

    def test_rectangular_grid(self):
        n, m = 10, 7
        x = jnp.asarray(chebpts(n, kind=2))
        y = jnp.asarray(chebpts(m, kind=2))
        xx, yy = jnp.meshgrid(x, y)
        assert xx.shape == (m, n)
        assert yy.shape == (m, n)
        # rows of xx are x; columns of yy are y
        assert float(jnp.max(jnp.abs(xx[0] - x))) < TOL
        assert float(jnp.max(jnp.abs(yy[:, 0] - y))) < TOL
