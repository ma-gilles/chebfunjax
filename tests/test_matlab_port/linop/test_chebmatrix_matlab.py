"""Port of MATLAB Chebfun tests/linop/test_chebmatrix.m (Fable 5).

Uses the chebcolloc2 discretization, as the MATLAB test does.

Provenance
----------
MATLAB source : tests/linop/test_chebmatrix.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    I,
    eval_at,
    mult,
    sum_functional,
    zeros_op,
)
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.utils.quadrature import chebweights

jax.config.update("jax_enable_x64", True)

D5 = np.array([
    [-5.499999999999999, 6.828427124746189, -2.000000000000000,
     1.171572875253810, -0.500000000000000],
    [-1.707106781186547, 0.707106781186547, 1.414213562373095,
     -0.707106781186548, 0.292893218813452],
    [0.500000000000000, -1.414213562373095, 0.0,
     1.414213562373095, -0.500000000000000],
    [-0.292893218813452, 0.707106781186548, -1.414213562373095,
     -0.707106781186547, 1.707106781186547],
    [0.500000000000000, -1.171572875253810, 2.000000000000000,
     -6.828427124746189, 5.499999999999999],
])


def _blkdiag(mats):
    out = np.zeros((sum(m.shape[0] for m in mats),
                    sum(m.shape[1] for m in mats)))
    r = c = 0
    for m in mats:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]
        c += m.shape[1]
    return out


class TestLinopChebmatrix:
    def test_all_matlab_assertions(self):
        dom = (-2.0, -0.5, 1.0, 2.0)
        Id = I(dom)
        Dop = D(dom)
        Z = zeros_op(dom)
        x = cj.chebfun(lambda t: t, domain=dom)
        u = (x ** 2).sin()
        U = mult(u)
        n = [5, 5, 5]

        DD = _blkdiag([2 / 1.5 * D5, 2 / 1.5 * D5, 2 / 1.0 * D5])
        xx = np.asarray(ChebColloc2Disc(n, dom).points())
        ww = np.concatenate([
            np.asarray(chebweights(5)) * 0.5 * (dom[k + 1] - dom[k])
            for k in range(3)])
        UU = np.diag(np.asarray(u(jnp.asarray(xx))))

        err = []

        A = ChebMatrix([[Id, Z], [Dop, U]])
        M = np.asarray(A.dense(n))
        expected = np.block([[np.eye(15), np.zeros((15, 15))], [DD, UU]])
        err.append(float(np.linalg.norm(M - expected)))

        one = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom)
        A = ChebMatrix([
            [Id, x, -3 * Id],
            [sum_functional(dom), 5.0, eval_at(dom[-1], dom)],
            [Dop, one, U],
        ])
        M = np.asarray(A.dense(n))
        eval_row = np.zeros(15)
        eval_row[-1] = 1.0
        MM = np.block([
            [np.eye(15), xx[:, None], -3 * np.eye(15)],
            [ww[None, :], np.array([[5.0]]), eval_row[None, :]],
            [DD, np.ones((15, 1)), UU],
        ])
        err.append(float(np.linalg.norm(M - MM)))

        # Application to an appropriate chebmatrix.
        v = ChebMatrix([[x.exp()], [math.pi], [x.cos()]])
        Av = A * v
        err.append(float(((v[0] + math.pi * x - 3 * v[2]) - Av[0]).norm()))
        err.append(abs(float(v[0].sum()) + 5 * v[1]
                       + float(v[2](jnp.asarray(dom[-1]))) - float(Av[1])))
        err.append(float(((v[0].diff() + math.pi + u * v[2]) - Av[2]).norm()))

        assert all(e < 1e-14 for e in err), err
