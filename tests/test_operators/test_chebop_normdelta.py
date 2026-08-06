"""``info.normDelta`` for nonlinear systems, and the Newton stopping test.

MATLAB's ``[u, v, info] = N\\[0; 0]`` reports ``info.normDelta``, the norm
of each accepted Newton update; ode-nonlin/BVPSystem plots it against the
iteration number. Three things were wrong for systems:

* the history was empty -- only the scalar solver recorded it;
* the loop's only stopping test compared ``max|R|`` on the residual,
  whose derivative rows carry an n^2 scaling, so it ran on at
  machine-precision noise long after the iterate stopped moving;
* the reported value was the Euclidean norm of the discrete coefficient
  vector rather than MATLAB's chebfun (L2 function) norm, which is
  smaller by roughly sqrt(n).
"""
from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop


def _bvp_system():
    # ode-nonlin/BVPSystem: u'' = sin(v), v'' = -cos(u) on [-1, 1]
    N = Chebop(lambda x, u, v: [u.diff(2) - v.sin(), v.diff(2) + u.cos()],
               domain=(-1, 1))
    N.lbc = lambda u, v: [u - 1, v.diff()]
    N.rbc = lambda u, v: [v, u.diff()]
    return N


class TestNormDelta:
    def test_history_is_recorded_for_a_system(self):
        _sol, info = _bvp_system().solvebvp([0.0, 0.0])
        assert len(info["normDelta"]) > 0

    def test_history_converges_quadratically(self):
        _sol, info = _bvp_system().solvebvp([0.0, 0.0])
        d = info["normDelta"]
        # strictly decreasing, and the tail squares each step
        assert all(d[i + 1] < d[i] for i in range(len(d) - 1))
        assert d[-1] < 1e-12
        # quadratic: the last few drops are far steeper than linear
        assert d[-2] < d[-3] ** 1.5

    def test_no_trailing_noise_iterations(self):
        # The loop used to run on at machine precision: 15 entries with
        # the last 8 all ~1e-15.  Stopping on the update norm leaves a
        # single converged entry at the end.
        _sol, info = _bvp_system().solvebvp([0.0, 0.0])
        d = info["normDelta"]
        assert len(d) <= 10
        assert sum(1 for v in d if v < 1e-13) <= 1

    def test_norm_is_the_function_norm_not_the_vector_norm(self):
        # MATLAB's first update here is about 2.  The discrete vector
        # norm would be larger by roughly sqrt(n) (n = 22 -> ~11.5).
        _sol, info = _bvp_system().solvebvp([0.0, 0.0])
        assert 1.0 < info["normDelta"][0] < 4.0

    def test_solution_still_satisfies_every_boundary_condition(self):
        (u, v), _info = _bvp_system().solvebvp([0.0, 0.0])
        assert float(u(np.float64(-1.0))) == pytest.approx(1.0, abs=1e-10)
        assert float(v(np.float64(1.0))) == pytest.approx(0.0, abs=1e-10)
        assert float(u.diff()(np.float64(1.0))) == pytest.approx(
            0.0, abs=1e-9)
        assert float(v.diff()(np.float64(-1.0))) == pytest.approx(
            0.0, abs=1e-9)

    def test_linear_system_reports_an_empty_history(self):
        # No Newton iteration, so nothing to report -- as in MATLAB.
        N = Chebop(lambda x, u, v: [u.diff(2) - v, v.diff(2) + u],
                   domain=(-1, 1))
        N.lbc = lambda u, v: [u - 1, v.diff()]
        N.rbc = lambda u, v: [v, u.diff()]
        _sol, info = N.solvebvp([0.0, 0.0])
        assert info["normDelta"] == []
