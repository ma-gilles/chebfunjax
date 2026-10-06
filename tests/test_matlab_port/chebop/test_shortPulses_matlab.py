"""Port of MATLAB Chebfun tests/chebop/test_shortPulses.m (Fable 5).

Literal pass(3) compares the no-pulse solution with restart-off output.
Fresh pinned MATLAB confirms this predicate. Integrator differences are
reported as failures, not repaired by reversing the assertion operands.

Provenance
----------
MATLAB source : tests/chebop/test_shortPulses.m (chebfun #1512)
Chebfun commit: 7574c77
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopShortPulses:
    def test_all_matlab_assertions(self):
        t = chebfun(lambda s: s, domain=(0.0, 2.0))
        L = Chebop(lambda t: t.diff() + t, (0.0, 2.0), 1.0, None)
        u_no_pulse = L.solve(0.0)

        # %% Long pulses
        long_pulse = 20.0 * (t > 1.0) * (t < 1.2)
        u_long = L.solve(long_pulse)
        assert np.array_equal(u_long.domain.breakpoints,
                           (0.0, 1.0, 1.2, 2.0))          # pass(1)
        assert float((u_no_pulse - u_long).norm(2)) > 0.1

        # %% Short pulses
        short_pulse = 20.0 * (t > 1.0) * (t < 1.05)
        u_short = L.solve(short_pulse)
        assert np.array_equal(u_short.domain.breakpoints,
                           (0.0, 1.0, 1.05, 2.0))         # pass(2)
        assert float((u_no_pulse - u_short).norm(2)) > 0.1

        # %% Literal source restart-off predicate (the marcher misses the pulse).
        L.ivp_restart_solver = False
        u_no_restart = L.solve(short_pulse)
        assert np.array_equal(u_short.domain.breakpoints,
                           (0.0, 1.0, 1.05, 2.0))         # pass(3)
        assert float((u_no_pulse - u_no_restart).norm(2)) < 1e-10
