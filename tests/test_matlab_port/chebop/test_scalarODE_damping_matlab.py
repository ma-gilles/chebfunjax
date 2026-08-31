"""Port of MATLAB Chebfun tests/chebop/test_scalarODE_damping.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE_damping.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.operators.chebop import Chebop  # noqa: E402


def _make_op():
    N = Chebop(lambda x, u: 0.05 * u.diff(2) + (5.0 * x).cos() * u.sin(),
               domain=(0.0, math.pi))
    N.lbc = lambda u: u - 2.0
    N.rbc = lambda u: u - 3.0
    return N


class TestChebopScalarOdeDamping:
    def test_all_matlab_assertions(self):
        tol = 1e-9

        # %% chebcolloc2 (default collocation path)
        N = _make_op()
        u1 = N.solvebvp(0.0)[0]
        assert float(N(u1).norm(2)) < tol      # err(1)

        # %% chebcolloc1
        N = _make_op()
        u2 = N.solve(0.0, discretization="chebcolloc1")
        assert float(N(u2).norm(2)) < tol      # err(2)

        # %% ultraS
        N = _make_op()
        u3 = N.solve(0.0, discretization="ultraS")
        assert float(N(u3).norm(2)) < tol      # err(3)
