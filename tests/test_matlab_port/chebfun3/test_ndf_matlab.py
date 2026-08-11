"""Port of MATLAB Chebfun tests/chebfun3/test_ndf.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.ndf()`` and
``Chebfun3.length()`` now exist.

Provenance
----------
MATLAB source : tests/chebfun3/test_ndf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3


class TestChebfun3Ndf:
    def test_empty(self):
        # pass(1): ndf(chebfun3()) == 0.
        assert Chebfun3.empty().ndf() == 0

    def test_constant(self):
        # pass(2): a constant has rank (1,1,1) and degree 1 in each
        # direction, so ndf == 3*1 + 1 == 4.
        assert chebfun3(lambda x, y, z: 10.0 + 0 * x).ndf() == 4

    def test_linear_in_x(self):
        # pass(3): f = x on a non-default box.
        f = chebfun3(lambda x, y, z: x,
                     domain=(-1.0, 2.0, -np.pi / 2, np.pi, -3.0, 1.0))
        assert f.ndf() == 5

    def test_linear_in_y(self):
        # pass(4)
        assert chebfun3(lambda x, y, z: y).ndf() == 5

    def test_linear_in_z(self):
        # pass(5)
        f = chebfun3(lambda x, y, z: z,
                     domain=(-1.0, 2.0, -np.pi / 2, np.pi, -3.0, 1.0))
        assert f.ndf() == 5

    def test_matches_rank_times_length_formula(self):
        # pass(6): ndf == rX*m + rY*n + rZ*p + rX*rY*rZ.
        f = chebfun3(lambda x, y, z: np.pi * 0 + (x + y + z))
        rx, ry, rz = f.rank
        m, n, p = f.length()
        assert f.ndf() == rx * m + ry * n + rz * p + rx * ry * rz
