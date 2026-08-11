"""Port of MATLAB Chebfun tests/linop/test_systemapply.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_systemapply.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax

import chebfunjax as cj
from chebfunjax.operators.blocks import primitive_operators
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

EPS = 2.220446049250313e-16
TOL = 1e-14


class TestLinopSystemApply:
    def test_all_matlab_assertions(self):
        d = (-math.pi, math.pi)
        Z, I, D, C, M = primitive_operators(d)
        x = cj.chebfun(lambda t: t, domain=d)

        A = ChebMatrix([[I + 2 * D ** 2, -D], [D, Z]])
        B = A ** 2
        u = ChebMatrix([[x.sin()], [x.exp()]])

        v = A * u
        w = A * v
        z = (B + 3 * B.identity()) * u

        u1, u2 = u[0], u[1]

        assert float((v[0] - (u1 + 2 * u1.diff(2) - u2.diff())).norm()) \
            < 50 * TOL
        assert float((v[1] - u1.diff()).norm()) < 2 * TOL

        r = w - B * u
        assert all(float(blk.norm()) < 1e-10 * (TOL / EPS)
                   for row in r.blocks for blk in row)

        r = z - w - 3 * u
        assert all(float(blk.norm()) < 1e-10 * (TOL / EPS)
                   for row in r.blocks for blk in row)
