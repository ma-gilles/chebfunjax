"""Port of MATLAB Chebfun tests/ballfunv/test_PTdecomposition.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_PTdecomposition.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

# tol = 1e5 * pref.techPrefs.chebfuneps  (chebfuneps = machine eps)
TOL = 1e5 * float(np.finfo(np.float64).eps)


class TestBallfunvPtdecomposition:
    def test_all_matlab_assertions(self):
        pass_ = {}

        # Example 1 (spherical coordinates)
        p = Ballfun.from_function(
            lambda r, lam, th: np.cos(r**2 * np.sin(th)**2 * np.cos(lam) * np.sin(lam)),
            spherical=True,
        )
        t = Ballfun.from_function(
            lambda r, lam, th: np.sin(r**2 * np.sin(th) * np.cos(th) * np.sin(lam)),
            spherical=True,
        )
        V = Ballfunv.PT2ballfunv(p, t)
        p2, t2 = V.PTdecomposition()
        pass_[1] = (p.diff(2, 1, "spherical") - p2.diff(2, 1, "spherical")).norm() < TOL
        pass_[2] = (p.diff(3, 1, "spherical") - p2.diff(3, 1, "spherical")).norm() < TOL
        pass_[3] = (t.diff(2, 1, "spherical") - t2.diff(2, 1, "spherical")).norm() < TOL
        pass_[4] = (t.diff(3, 1, "spherical") - t2.diff(3, 1, "spherical")).norm() < TOL

        # Example 2 (Cartesian coordinates)
        p = Ballfun.from_function(lambda x, y, z: x**2 + y * z)
        t = Ballfun.from_function(lambda x, y, z: x * y * z)
        V = Ballfunv.PT2ballfunv(p, t)
        p2, t2 = V.PTdecomposition()
        pass_[5] = (p.diff(2, 1, "spherical") - p2.diff(2, 1, "spherical")).norm() < TOL
        pass_[6] = (p.diff(3, 1, "spherical") - p2.diff(3, 1, "spherical")).norm() < TOL
        pass_[7] = (t.diff(2, 1, "spherical") - t2.diff(2, 1, "spherical")).norm() < TOL
        pass_[8] = (t.diff(3, 1, "spherical") - t2.diff(3, 1, "spherical")).norm() < TOL

        for i in range(1, 9):
            assert pass_[i], f"PTdecomposition assertion {i} failed"
