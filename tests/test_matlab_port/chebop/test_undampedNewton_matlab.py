"""Port of MATLAB Chebfun tests/chebop/test_undampedNewton.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_undampedNewton.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopUndampednewton:
    def test_all_matlab_assertions(self):
        # Steady-state Allen-Cahn with damping turned off
        dom = (0.0, 10.0)
        f = chebfun(lambda x: x, domain=dom).sin()
        N = Chebop(lambda u: u.diff(2) + u - u ** 3, dom, 1.0, -1.0)
        N.damping = False  # pref.damping = 0
        u, _info = N.solvebvp(f)
        err = float((N(u) - f).norm(2))
        assert err < 1e-8
