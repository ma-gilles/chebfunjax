"""Port of MATLAB Chebfun tests/chebop/test_linearSystem2.m (Fable 5).

2x2 linear system in MATLAB's chebmatrix ``u{k}`` cell syntax, on a
smooth domain, a piecewise domain, and (function-handle op + ultraS)
with ``numVars``.  In the port the cell style is detected by probing,
so ``numVars`` is advisory.

Provenance
----------
MATLAB source : tests/chebop/test_linearSystem2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun, jump  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402

TOL = 1e-10
D = (-1.0, 1.0)


def _myop(x, u):
    return [u[0].diff() + u[0] + 2.0 * u[1],
            u[0].diff() - u[0] + u[1].diff()]


def _rhs():
    x = chebfun(lambda s: s, domain=D)
    return [x.exp(), chebfun(lambda s: 0.0 * s + 1.0, domain=D)]


def _check(A, u, f):
    res = A(u)
    err1 = math.sqrt(sum(
        float((r - g).norm(2)) ** 2 for r, g in zip(res, f)))
    bc_left = A.lbc(*u)
    bc_right = A.rbc(*u)
    errl = abs(float(bc_left(jnp.asarray(D[0]))))
    errr = abs(float(bc_right(jnp.asarray(D[1]))))
    return err1, errl, errr


class TestChebopLinearSystem2:
    def test_smooth_domain(self):
        A = Chebop(lambda x, u: [u[0].diff() + u[0] + 2.0 * u[1],
                                 u[0].diff() - u[0] + u[1].diff()], D)
        A.lbc = lambda u: u[0] + u[0].diff()
        A.rbc = lambda u: u[1].diff()
        f = _rhs()
        u = A.solve(f)
        err1, errl, errr = _check(A, u, f)
        assert err1 < TOL                                   # pass(1)
        assert errl < TOL and errr < TOL                    # pass(2)

    def test_piecewise_domain(self):
        A = Chebop(lambda x, u: [u[0].diff() + u[0] + 2.0 * u[1],
                                 u[0].diff() - u[0] + u[1].diff()],
                   (-1.0, 0.0, 1.0))
        A.lbc = lambda u: u[0] + u[0].diff()
        A.rbc = lambda u: u[1].diff()
        f = _rhs()
        u = A.solve(f)
        err3, errl, errr = _check(A, u, f)
        assert err3 < TOL                                   # pass(3)
        assert errl < TOL and errr < TOL                    # pass(4)
        assert abs(float(jump(u[0], 0.0))) < TOL            # pass(5)
        assert abs(float(jump(u[1], 0.0))) < TOL

    def test_function_handle_ultras(self):
        A = Chebop(_myop, D)
        A.lbc = lambda u: u[0] + u[0].diff()
        A.rbc = lambda u: u[1].diff()
        A.numVars = 2
        f = _rhs()
        u = A.solve(f, discretization="ultraS")
        err6, errl, errr = _check(A, u, f)
        assert err6 < TOL                                   # pass(6)
        assert errl < TOL and errr < TOL                    # pass(7)
