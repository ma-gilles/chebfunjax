"""Port of MATLAB Chebfun tests/chebop/test_ivp_chebmatrix_syntax.m (Fable 5).

MATLAB's chebmatrix cell syntax ``u{k}`` maps to ``u[k-1]`` on a single
indexable argument; chebop detects it by probing (see ``_cellify``).

Provenance
----------
MATLAB source : tests/chebop/test_ivp_chebmatrix_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.operators.chebop import Chebop  # noqa: E402

DOM = (0.0, 5.0)
TOL = 1e-14


def _end_vals(sol, x):
    return np.array([float(c(jnp.asarray(x))) for c in sol])


def _sys_diff(p, q):
    return max(float((a - b).norm(2)) for a, b in zip(p, q))


class TestChebopIvpChebmatrixSyntax:
    def test_all_matlab_assertions(self):
        # %% Reference: multiple-variable syntax (full Brusselator IVP)
        N = Chebop(lambda t, u, v, w: [
            u.diff() - 1.0 - u ** 2 * v + (w + 1.0) * u,
            v.diff() - u * w + u ** 2 * v,
            w.diff() - 1.5 + w * u], DOM)
        N.lbc = [1.0, 3.0, 4.0]
        uL = N.solve(0.0)
        assert np.linalg.norm(
            _end_vals(uL, DOM[0]) - [1.0, 3.0, 4.0]) < TOL   # pass(1)

        N.lbc = None
        N.rbc = [0.7, 0.9, 1.2]
        uR = N.solve(0.0)
        assert np.linalg.norm(
            _end_vals(uR, DOM[1]) - [0.7, 0.9, 1.2]) < TOL   # pass(2)

        # %% CHEBMATRIX syntax
        def make_m():
            return Chebop(lambda t, u: [
                u[0].diff() - 1.0 - u[0] ** 2 * u[1]
                + (u[2] + 1.0) * u[0],
                u[1].diff() - u[0] * u[2] + u[0] ** 2 * u[1],
                u[2].diff() - 1.5 + u[2] * u[0]], DOM)

        M = make_m()
        M.lbc = lambda u: [u[0] - 1.0, u[1] - 3.0, u[2] - 4.0]
        vL1 = M.solve(0.0)
        assert _sys_diff(vL1, uL) == 0.0                     # pass(3)

        M = make_m()
        M.lbc = [1.0, 3.0, 4.0]
        vL2 = M.solve(0.0)
        assert _sys_diff(vL2, uL) == 0.0                     # pass(4)

        M = make_m()
        M.rbc = lambda u: [u[0] - 0.7, u[1] - 0.9, u[2] - 1.2]
        vR1 = M.solve(0.0)
        assert _sys_diff(vR1, uR) == 0.0                     # pass(5)

        M = make_m()
        M.rbc = [0.7, 0.9, 1.2]
        vR2 = M.solve(0.0)
        assert _sys_diff(vR2, uR) == 0.0                     # pass(6)
