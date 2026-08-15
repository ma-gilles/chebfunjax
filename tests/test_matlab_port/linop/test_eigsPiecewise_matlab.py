"""Port of MATLAB Chebfun tests/linop/test_eigsPiecewise.m (Fable 5).

Ports the chebcolloc2 block (MATLAB err(3), err(4)).  The chebcolloc1 and
ultraS blocks are covered by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_eigsPiecewise.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    primitive_functionals,
    primitive_operators,
)

jax.config.update("jax_enable_x64", True)

V4_RESULTS = np.array([
    0.055633547864,
    0.058372224970,
    0.222492474974,
    0.233441822781,
    0.500447687842,
    0.525062886469,
])


class TestLinopEigsPiecewise:
    def test_all_matlab_assertions(self):
        d = (-5.0, 5.0)
        x = cj.chebfun(lambda t: t, domain=d)
        h = 0.1
        a = 4.0
        b = -1.0
        c = 0.9

        Z, I, diff_op, C, M = primitive_operators(d)
        z, e, s, r = primitive_functionals(d)

        op = -h * diff_op ** 2 + M(a * ((x - b).sign() - (x - c).sign()))
        L = linop(op).addbc(e(-5.0), 0.0).addbc(e(5.0), 0.0)

        lam, V = L.eigs(6, n=65)
        vals = np.asarray(lam).real
        err = [float(np.max(np.abs(vals - V4_RESULTS)))]

        resid = 0.0
        for j in range(6):
            v = V[j][0]
            resid = max(resid, float(abs((op * v - complex(lam[j]) * v)
                                         .norm())))
        err.append(resid)

        assert err[0] < 1e-10, err
        assert err[1] < 5e-8, err

    @pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
    def test_ultras_and_chebcolloc1(self, disc):
        # MATLAB err(1), err(2), err(5), err(6): the same piecewise
        # eigenproblem under ultraS and chebcolloc1.
        d = (-5.0, 5.0)
        x = cj.chebfun(lambda t: t, domain=d)
        h = 0.1
        a = 4.0
        b = -1.0
        c = 0.9

        Z, I, diff_op, C, M = primitive_operators(d)
        z, e, s, r = primitive_functionals(d)

        op = -h * diff_op ** 2 + M(a * ((x - b).sign()
                                        - (x - c).sign()))
        L = linop(op).addbc(e(-5.0), 0.0).addbc(e(5.0), 0.0)

        lam, V = L.eigs(6, n=65, discretization=disc)
        vals = np.sort(np.asarray(lam).real)
        assert float(np.max(np.abs(vals - V4_RESULTS))) < 1e-10

        # The recovered eigenfunctions are continuous across the
        # coefficient breakpoints only to solver accuracy, and
        # chebfunjax's diff() turns those ~1e-12 jumps into Dirac
        # deltas whose norm is infinite (MATLAB's diff does not).
        # Strip the deltas but assert their magnitude at the same
        # tolerance, so nothing is discarded silently.
        resid = 0.0
        for j in range(6):
            v = V[j][0]
            r = op * v - complex(lam[j]) * v
            deltas = getattr(r, "deltas", ())
            worst_delta = max((abs(complex(d[1])) for d in deltas),
                              default=0.0)
            r = cj.Chebfun(funs=r.funs, domain=r.domain)
            resid = max(resid, float(abs(r.norm())), worst_delta)
        assert resid < 5e-8, resid
