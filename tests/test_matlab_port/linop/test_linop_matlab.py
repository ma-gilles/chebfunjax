"""Port of MATLAB Chebfun tests/linop/test_linop.m (Fable 5).

Ports the chebcolloc2 pass of the MATLAB loop (k = 1).  The ultraS pass is
covered by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_linop.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, I, eval_at, sum_functional
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)


class TestLinopLinop:
    def test_all_matlab_assertions(self):
        dom = (-2.0, 2.0)
        Id = I(dom)
        Dop = D(dom)
        x = cj.chebfun(lambda t: t, domain=dom)

        L = linop(ChebMatrix([[Dop, -Id], [Id, Dop]]))
        f = [x, 0 * x]
        El = eval_at(dom[0], dom)
        Er = eval_at(dom[-1], dom)
        L = L.addbc([El, -Er], 0.0)
        L = L.addbc([sum_functional(dom), El], 1.0)

        u = L.linsolve(f)
        u1, u2 = u[0], u[1]

        err = []
        # Check the ODEs.
        err.append(float((u1.diff() - u2 - f[0]).norm()))
        err.append(float((u1 + u2.diff()).norm()))
        # Check the BCs.
        err.append(abs(float(u1(jnp.asarray(-2.0)))
                       - float(u2(jnp.asarray(2.0)))))
        err.append(abs(float(u1.sum()) + float(u2(jnp.asarray(-2.0))) - 1.0))

        assert all(e < 1e-9 for e in err), err

    def test_ultras(self):
        # MATLAB's k = 2 pass: the same first-order system under the
        # ultraS discretization.
        dom = (-2.0, 2.0)
        Id = I(dom)
        Dop = D(dom)
        x = cj.chebfun(lambda t: t, domain=dom)

        L = linop(ChebMatrix([[Dop, -Id], [Id, Dop]]))
        f = [x, 0 * x]
        El = eval_at(dom[0], dom)
        Er = eval_at(dom[-1], dom)
        L = L.addbc([El, -Er], 0.0)
        L = L.addbc([sum_functional(dom), El], 1.0)

        u = L.linsolve(f, n=64, discretization="ultraS")
        u1, u2 = u[0], u[1]

        err = []
        err.append(float((u1.diff() - u2 - f[0]).norm()))
        err.append(float((u1 + u2.diff()).norm()))
        err.append(abs(float(u1(jnp.asarray(-2.0)))
                       - float(u2(jnp.asarray(2.0)))))
        err.append(abs(float(u1.sum())
                       + float(u2(jnp.asarray(-2.0))) - 1.0))
        assert all(e < 1e-9 for e in err), err
