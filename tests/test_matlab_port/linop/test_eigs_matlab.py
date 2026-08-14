"""Port of MATLAB Chebfun tests/linop/test_eigs.m (Fable 5).

Ports the chebcolloc2 block of the MATLAB test.  The chebcolloc1 and ultraS
blocks are covered by a separate skipped test: chebfunjax's BlockLinop only
implements the chebcolloc2 (rectangular collocation) discretization.

Provenance
----------
MATLAB source : tests/linop/test_eigs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, eval_at

jax.config.update("jax_enable_x64", True)

TOL = 1e-7


def _build():
    dom = (-math.pi / 2, math.pi / 2)
    D2 = D(dom, 2)
    L = linop(D2).addbc(eval_at(dom[0], dom), 0.0)
    L = L.addbc(eval_at(dom[-1], dom), 0.0)
    return dom, D2, L


class TestLinopEigs:
    def test_all_matlab_assertions(self):
        dom, D2, L = _build()
        e_true = -np.arange(1, 7)[::-1].astype(float) ** 2

        lam, V = L.eigs(6, n=65)
        e = np.asarray(lam).real
        err = [float(np.max(np.abs(e - e_true)))]

        # Check that we actually computed eigenfunctions.
        resid = 0.0
        for j in range(6):
            v = V[j][0]
            resid = max(resid, float(abs((D2 * v - complex(lam[j]) * v)
                                         .norm())))
        err.append(resid)

        assert all(x < TOL for x in err), err

    def test_ultras_and_chebcolloc1(self):
        # MATLAB err(3)-err(6): the same eigenproblem under the ultraS
        # and chebcolloc1 discretizations.
        dom, D2, L = _build()
        e_true = -np.arange(1, 7)[::-1].astype(float) ** 2
        for disc in ("ultraS", "chebcolloc1"):
            lam, V = L.eigs(6, 0, n=65, discretization=disc)
            e = np.sort(np.asarray(lam).real)
            assert float(np.max(np.abs(e - e_true))) < 1e-8, disc
            resid = 0.0
            for j in range(6):
                v = V[j][0]
                resid = max(resid, float(abs(
                    (D2 * v - complex(np.asarray(lam)[j]) * v).norm())))
            assert resid < 1e-8, (disc, resid)
        # Scalar solves under both backends (the linsolve halves of the
        # MATLAB discretization sweep): u'' = 1, u(+-1) = 0.
        from chebfunjax.operators.blocklinop import linop as _linop
        dom1 = (-1.0, 1.0)
        Ls = _linop(D(dom1, 2)).addbc(eval_at(-1.0, dom1)).addbc(
            eval_at(1.0, dom1))
        for disc in ("ultraS", "chebcolloc1"):
            u = Ls.linsolve(1.0, n=64, discretization=disc)[0]
            assert abs(float(u(jnp.asarray(0.0))) + 0.5) < 1e-10, disc
