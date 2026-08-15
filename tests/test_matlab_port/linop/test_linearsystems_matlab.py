"""Port of MATLAB Chebfun tests/linop/test_linearsystems.m (Fable 5).

Ports the chebcolloc2 passes of the MATLAB loop (k = 1 with the smooth RHS
and k = 4 with the RHS carrying a breakpoint).  The ultraS and chebcolloc1
passes are covered by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_linearsystems.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    D,
    I,
    eval_at,
    mult,
    sum_functional,
    zero_functional,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


class TestLinopLinearSystems:
    def test_all_matlab_assertions(self):
        dom = (-1.0, 1.0)
        Id = I(dom)
        Dop = D(dom)
        x = cj.chebfun(lambda t: t, domain=dom)
        c = (x ** 2).sin()
        C = mult(c)
        El = eval_at(dom[0], dom)
        Er = eval_at(dom[-1], dom)
        z = zero_functional(dom)
        zero = cj.chebfun(lambda t: jnp.zeros_like(t), domain=dom)

        L = linop(ChebMatrix([
            [Dop ** 2, -Id, x.sin()],
            [C, Dop, zero],
            [z, El, 4.0],
        ]))
        L = L.addbc([El, -Er, 0.0], 0.0)
        L = L.addbc([sum_functional(dom), El, 0.0], 1.0)
        L = L.addbc([Er * Dop, z, 0.0], 0.0)

        for f in ([x - 1, zero, 1.0], [abs(x - 1), 0 * x, 1.0]):
            w = L.linsolve(f)
            w1, w2, w3 = w[0], w[1], w[2]
            f1 = f[0]

            err = []
            # Check the ODEs.
            err.append(float((w1.diff(2) - w2 + x.sin() * w3 - f1).norm()))
            err.append(float((c * w1 + w2.diff()).norm()))
            err.append(abs(float(w2(jnp.asarray(dom[0]))) + 4 * w3 - 1.0))
            # Check the BCs.
            err.append(abs(float(w1(jnp.asarray(dom[0])))
                           - float(w2(jnp.asarray(dom[-1])))))
            err.append(abs(float(w1.sum())
                           + float(w2(jnp.asarray(dom[0]))) - 1.0))
            err.append(abs(float(w1.diff()(jnp.asarray(dom[-1])))))
            # Check continuity.
            Du = w1.diff()
            for g in (w1, w2, Du):
                err.append(abs(float(g(jnp.asarray(1.0), "left"))
                               - float(g(jnp.asarray(1.0), "right"))))

            assert all(e < TOL for e in err), err

    @pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
    def test_ultras_and_chebcolloc1(self, disc):
        # MATLAB's k = 2, 3, 5, 6 passes: the same block-system solve
        # under the ultraS and chebcolloc1 discretizations.
        dom = (-1.0, 1.0)
        Id = I(dom)
        Dop = D(dom)
        x = cj.chebfun(lambda t: t, domain=dom)
        c = (x ** 2).sin()
        C = mult(c)
        El = eval_at(dom[0], dom)
        Er = eval_at(dom[-1], dom)
        z = zero_functional(dom)
        zero = cj.chebfun(lambda t: jnp.zeros_like(t), domain=dom)

        L = linop(ChebMatrix([
            [Dop ** 2, -Id, x.sin()],
            [C, Dop, zero],
            [z, El, 4.0],
        ]))
        L = L.addbc([El, -Er, 0.0], 0.0)
        L = L.addbc([sum_functional(dom), El, 0.0], 1.0)
        L = L.addbc([Er * Dop, z, 0.0], 0.0)

        for f in ([x - 1, zero, 1.0], [abs(x - 1), 0 * x, 1.0]):
            w = L.linsolve(f, n=64, discretization=disc)
            w1, w2, w3 = w[0], w[1], w[2]
            f1 = f[0]

            err = []
            err.append(float((w1.diff(2) - w2 + x.sin() * w3
                              - f1).norm()))
            err.append(float((c * w1 + w2.diff()).norm()))
            err.append(abs(float(w2(jnp.asarray(dom[0])))
                           + 4 * w3 - 1.0))
            err.append(abs(float(w1(jnp.asarray(dom[0])))
                           - float(w2(jnp.asarray(dom[-1])))))
            err.append(abs(float(w1.sum())
                           + float(w2(jnp.asarray(dom[0]))) - 1.0))
            err.append(abs(float(w1.diff()(jnp.asarray(dom[-1])))))
            assert all(e < TOL for e in err), err
