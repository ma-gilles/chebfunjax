"""Port of MATLAB Chebfun tests/linop/test_eigsGeneralizedSys.m (Fable 5).

Ports the chebcolloc2 block (MATLAB err(1,3), err(1,4)).  The chebcolloc1 and
ultraS blocks are covered by a separate skipped test.

Deviation from the MATLAB assertion: the twelve eigenvalues form six double
pairs, so which member of a pair LAPACK returns first is not reproducible
across implementations.  MATLAB's ``norm(e - e_true)`` is therefore replaced
by a nearest-neighbour match of the two multisets at the same tolerance.

Provenance
----------
MATLAB source : tests/linop/test_eigsGeneralizedSys.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, I, eval_at, zero_functional
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL_VALS = 6e-9
TOL_FUNS = 4e-7


class TestLinopEigsGeneralizedSys:
    def test_all_matlab_assertions(self):
        dom = (-1.0, 1.0)
        diff_op = D(dom)
        Id = I(dom)

        def ev(t):
            return eval_at(t, dom)

        z = zero_functional(dom)

        A = ChebMatrix([[diff_op ** 2, diff_op], [diff_op, diff_op ** 2]])
        B = ChebMatrix([[Id, diff_op], [diff_op, Id]])
        A = linop(A)
        A = A.addbc([ev(-1.0), z]).addbc([ev(1.0), z])
        A = A.addbc([z, ev(-1.0)]).addbc([z, ev(1.0)])
        B = linop(B)

        e_true = -1 + 1j * math.pi * np.array(
            [-1, 1, -1, 1, -2, 2, -2, 2, -3, 3, -3, 3], dtype=float)

        lam, V = A.eigs(12, 0, B=B, n=65)
        e = np.asarray(lam)

        # Multiset match (see the module docstring).
        remaining = list(e)
        worst = 0.0
        for target in e_true:
            k = int(np.argmin(np.abs(np.asarray(remaining) - target)))
            worst = max(worst, float(abs(remaining.pop(k) - target)))
        assert worst < 10 * TOL_VALS, worst

        resid = 0.0
        for j in range(12):
            v1, v2 = V[j][0], V[j][1]
            lj = complex(e[j])
            resid = max(
                resid,
                float(abs((v1.diff(2) + v2.diff()
                           - (v1 + v2.diff()) * lj).norm())),
                float(abs((v1.diff() + v2.diff(2)
                           - (v1.diff() + v2) * lj).norm())))
        assert resid < 10 * TOL_FUNS, resid

    @pytest.mark.skip(
        reason="MATLAB err(1,1), err(1,2), err(1,5), err(1,6) repeat the "
               "problem with the chebcolloc1 and ultraS discretizations; "
               "chebfunjax's BlockLinop only implements chebcolloc2 "
               "rectangular collocation.")
    def test_ultras_and_chebcolloc1(self):
        raise NotImplementedError
