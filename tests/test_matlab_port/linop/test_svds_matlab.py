"""Port of MATLAB Chebfun tests/linop/test_svds.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_svds.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, eval_at
from chebfunjax.operators.linop_adjoint import svds

jax.config.update("jax_enable_x64", True)


def _cols(M):
    return [b for row in M.blocks for b in row] if hasattr(M, "blocks") else [M]


def _resid(L, U, S, V):
    tot = 0.0
    for i in range(len(V)):
        r = _cols(L * V[i])[0] - U[i] * float(S[i, i])
        tot += float(np.real(np.asarray(r.norm(2)))) ** 2
    return np.sqrt(tot)


class TestLinopSvds:
    def test_all_matlab_assertions(self):
        tol = 1e1 * 5e-13
        dom = (0.0, np.pi)
        Dd = D(dom)
        El = eval_at(dom[0], dom)
        Er = eval_at(dom[1], dom)
        s_true = np.arange(5, -1, -1, dtype=float)
        L = linop(Dd)
        U, S, V = svds(L, 6, "bvp")
        s = np.diag(S)
        assert np.max(np.abs(s - s_true)) < np.max(s) * tol                 # pass(1)
        assert _resid(L, U, S, V) < np.max(s) * tol                          # pass(2)
        s_true = np.arange(6, 0, -1, dtype=float) ** 2
        L = linop(Dd ** 2).addbc(El, 0.0).addbc(Er, 0.0)
        U, S, V = svds(L, 6, "bvp")
        s = np.diag(S)
        assert np.max(np.abs(s - s_true)) < np.max(s) * tol                 # pass(3)
        assert _resid(L, U, S, V) < np.max(s) * tol                          # pass(4)
