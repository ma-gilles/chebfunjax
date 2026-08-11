"""Port of MATLAB Chebfun tests/linop/test_linjump.m (Fable 5).

Ports the chebcolloc2 pass (MATLAB k = 1).  The chebcolloc1 and ultraS
passes are covered by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_linjump.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    jump_at,
    jump_functional,
    primitive_functionals,
    primitive_operators,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


def _strip_deltas(f):
    """Drop the Dirac part of a residual (MATLAB ``res.funs{k}.funPart``)."""
    return Chebfun(funs=f.funs, domain=f.domain)


class TestLinopLinJump:
    def test_all_matlab_assertions(self):
        dom = (0.0, 0.3, 1.0)
        Z, I, Dop, C, M = primitive_operators(dom)
        z, e, s, dt = primitive_functionals(dom)
        j = jump_at(dom)

        A = linop(ChebMatrix([[Dop ** 2, I], [-Dop, Dop ** 2 + I]]))

        A = A.add_constraint([e(0.0), z], -1.0)
        A = A.add_constraint([e(1.0), z], 0.0)
        A = A.add_constraint([z, e(0.0)], 0.0)
        A = A.add_constraint([z, e(1.0)], 1.0)
        A = A.add_continuity([j(0.3, 1), z], 2.0)
        A = A.add_continuity([z, j(0.3, 1)], 0.0)
        A = A.add_continuity([j(0.3, 0), z], 0.0)
        A = A.add_continuity([z, j(0.3, 0)], 1.0)

        x = cj.chebfun(lambda t: t, domain=dom)
        u = A.linsolve([x, 0 * x])
        u1, u2 = u[0], u[1]

        err = []
        # Jumps.
        J = jump_functional(0.3, dom, 0)
        err.append(abs(float(J * u2) - 1.0))
        err.append(abs(float(J * (Dop * u1)) - 2.0))
        # BCs.
        err.append(abs(float(u1(jnp.asarray(0.0))) + 1.0))
        err.append(abs(float(u2(jnp.asarray(1.0))) - 1.0))
        # ODEs (deltafuns stripped, exactly as the MATLAB test does).
        err.append(float(_strip_deltas(u1.diff(2) + u2 - x).norm()))
        err.append(float(_strip_deltas(-u1.diff() + u2.diff(2) + u2).norm()))

        assert all(v < TOL for v in err), err

    @pytest.mark.skip(
        reason="MATLAB's k = 2, 3 passes repeat the solve with the "
               "chebcolloc1 and ultraS discretizations; chebfunjax's "
               "BlockLinop only implements chebcolloc2 rectangular "
               "collocation.")
    def test_ultras_and_chebcolloc1(self):
        raise NotImplementedError
