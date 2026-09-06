"""Port of MATLAB Chebfun tests/chebop/test_nonlinSys2_C2.m (Fable 5).

Same system as test_nonlinSystem1 but written in MATLAB's chebmatrix
``u{k}`` cell syntax (``u[k-1]`` here, detected by probing).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSys2_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopNonlinSys2C2:
    def test_all_matlab_assertions(self):
        tol = 1e-10
        d = (-math.pi, math.pi)

        A = Chebop(lambda x, u: [u[0] - u[1].diff(2) + u[0] ** 2,
                                 u[0].diff() + u[1].sin()], d)
        A.lbc = lambda u: u[0] - 1.0
        A.rbc = lambda u: [u[1] - 0.5, u[1].diff()]

        u = A.solve([0.0, 0.0])

        # residual (err1)
        res = A(u)
        err1 = math.sqrt(sum(float(r.norm(2)) ** 2 for r in res))
        assert err1 < tol                                   # pass(1)

        # boundary conditions (err2): the stored bcs are the expanded
        # fixed-arity forms, so call them with unpacked components.
        bc_left = A.lbc(*u)
        bc_right = A.rbc(*u)
        if not isinstance(bc_left, (list, tuple)):
            bc_left = [bc_left]
        errl = max(abs(float(g(jnp.asarray(d[0])))) for g in bc_left)
        errr = max(abs(float(g(jnp.asarray(d[1])))) for g in bc_right)
        assert errl < tol and errr < tol                    # pass(2)
