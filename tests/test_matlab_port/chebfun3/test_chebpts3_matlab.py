"""Port of MATLAB Chebfun tests/chebfun3/test_chebpts3.m (Fable 5).

MATLAB chebpts3 builds ndgrid tensors of Chebyshev points; the port
builds the same from 1-D chebpts and checks the same identities.

Provenance
----------
MATLAB source : tests/chebfun3/test_chebpts3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)


class TestChebfun3Chebpts3:
    def test_cube_grid(self):
        n = 7
        x = np.asarray(chebpts(n, kind=2))
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        assert xx.shape == (n, n, n)
        assert float(np.max(np.abs(xx[:, 0, 0] - x))) < 10 * EPS
        assert float(np.max(np.abs(yy[0, :, 0] - x))) < 10 * EPS
        assert float(np.max(np.abs(zz[0, 0, :] - x))) < 10 * EPS
