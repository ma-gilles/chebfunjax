"""Port of MATLAB Chebfun tests/linop/test_eigsGeneralized.m (Fable 5).

Ports the chebcolloc2 blocks (MATLAB err(1,3), err(1,4), err(2,3), err(2,4)).
The chebcolloc1 and ultraS blocks are covered by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_eigsGeneralized.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, eval_at

jax.config.update("jax_enable_x64", True)

TOL_VALS = 6e-9
TOL_FUNS = 4e-7


class TestLinopEigsGeneralized:
    def test_all_matlab_assertions(self):
        dom = (-1.0, 1.0)
        diff_op = D(dom)

        def ev(t):
            return eval_at(t, dom)

        # D^2*u = 1i*lam*D*u,  u(-1) = u(1) = 0.
        A = linop(diff_op ** 2).addbc(ev(-1.0)).addbc(ev(1.0))
        B = linop(1j * diff_op)
        e_true = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])

        lam, V = A.eigs(6, 0, B=B, n=65)
        e = np.asarray(lam) / math.pi
        err = float(np.linalg.norm(np.sort(e.real) - e_true)
                    + np.linalg.norm(e.imag))
        assert err < 10 * TOL_VALS, err

        resid = 0.0
        for j in range(6):
            v = V[j][0]
            resid = max(resid, float(abs(
                (v.diff(2) - (1j * v.diff()) * complex(lam[j])).norm())))
        assert resid < 10 * TOL_FUNS, resid

        # Now with the highest derivative on the right:
        # 1i*D*u = (1/lam)*D^2*u,  u(-1) = u(1) = 0.
        A = linop(diff_op ** 2)
        B = linop(1j * diff_op).addbc(ev(-1.0)).addbc(ev(1.0))
        e_true = np.arange(1.0, 7.0)

        lam, V = B.eigs(6, 1, B=A, n=65)
        e = 1.0 / np.asarray(lam) / math.pi
        err = float(np.linalg.norm(np.sort(e.real) - e_true)
                    + np.linalg.norm(e.imag))
        assert err < 10 * TOL_VALS, err

        resid = 0.0
        for j in range(6):
            v = V[j][0]
            resid = max(resid, float(abs(
                (1j * v.diff() - v.diff(2) * complex(lam[j])).norm())))
        assert resid < 10 * TOL_FUNS, resid

    @pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
    def test_ultras_and_chebcolloc1(self, disc):
        # MATLAB err(:,1), err(:,2), err(:,5), err(:,6): the same
        # generalized problems under ultraS and chebcolloc1.
        dom = (-1.0, 1.0)
        diff_op = D(dom)

        def ev(t):
            return eval_at(t, dom)

        A = linop(diff_op ** 2).addbc(ev(-1.0)).addbc(ev(1.0))
        B = linop(1j * diff_op)
        e_true = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])

        lam, V = A.eigs(6, 0, B=B, n=65, discretization=disc)
        e = np.asarray(lam) / math.pi
        err = float(np.linalg.norm(np.sort(e.real) - e_true)
                    + np.linalg.norm(e.imag))
        assert err < 10 * TOL_VALS, err

        resid = 0.0
        for j in range(6):
            v = V[j][0]
            resid = max(resid, float(abs(
                (v.diff(2) - (1j * v.diff())
                 * complex(lam[j])).norm())))
        assert resid < 10 * TOL_FUNS, resid

        A = linop(diff_op ** 2)
        B = linop(1j * diff_op).addbc(ev(-1.0)).addbc(ev(1.0))
        e_true = np.arange(1.0, 7.0)

        lam, V = B.eigs(6, 1, B=A, n=65, discretization=disc)
        e = 1.0 / np.asarray(lam) / math.pi
        err = float(np.linalg.norm(np.sort(e.real) - e_true)
                    + np.linalg.norm(e.imag))
        assert err < 10 * TOL_VALS, err

        resid = 0.0
        for j in range(6):
            v = V[j][0]
            resid = max(resid, float(abs(
                (1j * v.diff() - v.diff(2)
                 * complex(lam[j])).norm())))
        assert resid < 10 * TOL_FUNS, resid
