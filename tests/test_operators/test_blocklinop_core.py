"""Core smoke tests for the BlockLinop layer (operators/blocklinop.py).

The MATLAB-parity coverage of this subsystem lives in
tests/test_matlab_port/linop/; these compact core tests exercise the
main paths (block algebra, application, block solve with constraints,
generalized eigenvalues, one-sided functionals) so the non-port CI job
also runs and covers them.

Provenance
----------
MATLAB source : @linop/linop.m, @chebmatrix/chebmatrix.m (behaviour
    pinned in detail by the tests/linop ports)
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators import primitive_operators
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import eval_at, mult, sum_functional
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

EPS = float(np.finfo(np.float64).eps)


class TestBlockAlgebraAndApply:
    def test_apply_and_power(self):
        d = (-math.pi, math.pi)
        Z, Id, D, C, M = primitive_operators(d)
        x = cj.chebfun(lambda t: t, domain=d)

        A = ChebMatrix([[Id + 2 * D ** 2, -D], [D, Z]])
        u = ChebMatrix([[x.sin()], [x.exp()]])
        v = A * u
        u1, u2 = u[0], u[1]
        assert float((v[0] - (u1 + 2 * u1.diff(2) - u2.diff())).norm()) \
            < 1e-11
        assert float((v[1] - u1.diff()).norm()) < 1e-12

        # (A^2 + 3I) u == A(Au) + 3u
        B = A ** 2
        w = A * v
        z = (B + 3 * B.identity()) * u
        r = z - w - 3 * u
        assert all(float(blk.norm()) < 1e-8
                   for row in r.blocks for blk in row)

    def test_operator_arith_on_chebfun(self):
        d = (-1.0, 4.0)
        Z, Id, D, C, M = primitive_operators(d)
        f = cj.chebfun(lambda t: jnp.exp(jnp.sin(t) ** 2 + 2), domain=d)
        A = -(2 * D ** 2 - M(f) * C + 3 * Id)
        Af = A * f
        want = f * f.cumsum() - 2 * f.diff(2) - 3 * f
        assert float((Af - want).norm()) < 1e4 * EPS * float(f.vscale)


class TestBlockSolve:
    def test_coupled_system_with_bcs(self):
        # Compact version of the tests/linop test_linearsystems port.
        from chebfunjax.operators.blocks import zero_functional

        dom = (-1.0, 1.0)
        Z, Id, D, C, M = primitive_operators(dom)
        x = cj.chebfun(lambda t: t, domain=dom)
        c = (x ** 2).sin()
        El, Er = eval_at(dom[0], dom), eval_at(dom[1], dom)
        z = zero_functional(dom)
        zero = cj.chebfun(lambda t: jnp.zeros_like(t), domain=dom)

        L = linop(ChebMatrix([
            [D ** 2, -Id, x.sin()],
            [mult(c), D, zero],
            [z, El, 4.0],
        ]))
        L = L.addbc([El, -Er, 0.0], 0.0)
        L = L.addbc([sum_functional(dom), El, 0.0], 1.0)
        L = L.addbc([Er * D, z, 0.0], 0.0)

        w = L.linsolve([x - 1, zero, 1.0])
        w1, w2, w3 = w[0], w[1], w[2]
        assert float((w1.diff(2) - w2 + x.sin() * w3 - (x - 1)).norm()) \
            < 1e-6
        assert float((c * w1 + w2.diff()).norm()) < 1e-6
        assert abs(float(w2(jnp.asarray(dom[0]))) + 4 * float(w3) - 1.0) \
            < 1e-8


class TestGeneralizedEigs:
    def test_first_derivative_pair(self):
        # Compact version of the tests/linop test_eigsGeneralized port:
        # D^2 u = 1i lam D u, u(+-1) = 0 -> lam/pi = -3..3 (excluding 0).
        from chebfunjax.operators.blocks import D as D_op

        dom = (-1.0, 1.0)
        diff_op = D_op(dom)
        A = linop(diff_op ** 2).addbc(eval_at(-1.0, dom)).addbc(
            eval_at(1.0, dom))
        B = linop(1j * diff_op)
        lam, _ = A.eigs(6, 0, B=B, n=65)
        e = np.sort((np.asarray(lam) / math.pi).real)
        want = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        assert np.max(np.abs(e - want)) < 1e-6
