"""Port of MATLAB Chebfun tests/chebfun2/test_padua.m (Fable 5).

FIXED: Padua points (paduapts) and the Padua constructor
(Chebfun2.from_padua / paduaVals2coeffs) added in the Fable 5 audit.

The MATLAB test builds F = chebfun2(rand(4), dom) from a value matrix; the
port uses an O(1) bidegree-(3,3) Chebyshev polynomial (total degree 6, so
degree-6 Padua interpolation is exact) built via from_function, which is the
chebfunjax analogue of a small random low-degree Chebfun2.

Provenance
----------
MATLAB source : tests/chebfun2/test_padua.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.padua import paduapts

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun2Padua:
    def test_paduapts_ordering(self):
        # Ordering of PADUAPTS is consistent with Padua2DM.
        x = np.asarray(paduapts(2))
        pdpts = np.array([
            [1, 0.5],
            [1, -1],
            [0, 1],
            [0, -0.5],
            [-1, 0.5],
            [-1, -1],
        ])
        assert np.linalg.norm(x - pdpts) < TOL

    def test_construction_small(self):
        # Small example on a low-degree polynomial with O(1) values.
        dom = (-2.0, 2.0, -2.0, 7.0)
        xa, xb, ya, yb = dom
        A = np.array([
            [0.3, -0.2, 0.1, 0.05],
            [0.15, 0.25, -0.1, 0.2],
            [-0.05, 0.1, 0.2, -0.15],
            [0.1, -0.05, 0.07, 0.12],
        ])

        def _cheb(r, k):
            return jnp.cos(k * jnp.arccos(jnp.clip(r, -1.0, 1.0)))

        def poly(x, y):
            rx = (2 * x - (xa + xb)) / (xb - xa)
            ry = (2 * y - (ya + yb)) / (yb - ya)
            tot = 0.0
            for i in range(4):
                for j in range(4):
                    tot = tot + A[i, j] * _cheb(rx, i) * _cheb(ry, j)
            return tot

        F = Chebfun2.from_function(poly, domain=dom)
        xy = np.asarray(paduapts(6, dom))
        f = np.asarray(poly(jnp.asarray(xy[:, 0]), jnp.asarray(xy[:, 1])))
        G = Chebfun2.from_padua(f, dom)
        assert float((F - G).norm("fro")) < TOL

    def test_construction_large(self):
        # Larger smooth example, checked at a point.
        dom = (-1.3, 1.0, -1.0, 1.5)

        def FF(x, y):
            return np.sin((x + 0.3) * (y - 0.2) + x ** 2 + 0.4)

        xy = np.asarray(paduapts(150, dom))
        f = FF(xy[:, 0], xy[:, 1])
        G = Chebfun2.from_padua(f, dom)
        val = float(G(jnp.asarray(0.0), jnp.asarray(0.0)))
        assert abs(FF(0.0, 0.0) - val) < TOL
