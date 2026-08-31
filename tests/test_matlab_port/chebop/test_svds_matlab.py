"""Port of MATLAB Chebfun tests/chebop/test_svds.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_svds.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopSvds:
    def test_all_matlab_assertions(self):
        # tol = 1e1 * pref.bvpTol (bvpTol default 1e-10)
        tol = 1e1 * 1e-10

        # %% SVD of differential operator: N = d/dx on [0, pi]
        N = Chebop(lambda x, u: u.diff(), domain=(0.0, math.pi))
        U, S, V = N.svds(10)
        s = np.diag(np.asarray(S))
        s_true = np.arange(9, -1, -1, dtype=float)
        # pass(1)
        assert np.linalg.norm(s - s_true, np.inf) < tol * s.max()
        # pass(2): norm(N(V) - U*S) over the quasimatrix columns
        sq = 0.0
        for j in range(10):
            d = N(V[j]) - float(s[j]) * U[j]
            sq += float(d.norm(2)) ** 2
        assert math.sqrt(sq) < tol * s.max()
