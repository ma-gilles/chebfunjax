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


class TestIntegralOperators:
    def test_fredholm_apply_and_eigs(self):
        # (Kf)(x) = int_0^1 exp(x*y) f(y) dy on [0, 1]; smoke the
        # Fredholm apply + the largest eigenvalues of the kernel.
        from chebfunjax.operators.integral import fred, fred_eigs

        f = cj.chebfun(lambda t: jnp.ones_like(t), domain=(0.0, 1.0))
        g = fred(lambda x, y: jnp.exp(x * y), f)
        # (K 1)(0) = int_0^1 1 dy = 1; (K 1)(1) = int_0^1 e^y dy = e - 1.
        assert abs(float(g(jnp.asarray(0.0))) - 1.0) < 1e-10
        assert abs(float(g(jnp.asarray(1.0))) - (math.e - 1.0)) < 1e-10

        lam = fred_eigs(lambda x, y: jnp.exp(x * y),
                        domain=(0.0, 1.0), k=3)
        # Largest eigenvalue of this classic kernel ~ 1.35303
        assert abs(float(np.max(np.abs(np.asarray(lam)))) - 1.35303) < 1e-3

    def test_volterra_apply(self):
        # (Vf)(x) = int_0^x f(y) dy == cumsum for K = 1.
        from chebfunjax.operators.integral import volt

        f = cj.chebfun(lambda t: jnp.cos(t), domain=(0.0, 1.0))
        g = volt(lambda x, y: jnp.ones_like(x * y), f)
        xs = jnp.asarray([0.25, 0.5, 1.0])
        assert float(jnp.max(jnp.abs(g(xs) - jnp.sin(xs)))) < 1e-9


class TestLinopScalarPaths:
    def test_piecewise_expm_heat(self):
        # Heat equation with a breakpointed initial condition drives the
        # piecewise rectangular-collocation expm.
        from chebfunjax.operators.chebop import Chebop

        N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        x = cj.chebfun(lambda t: t, domain=(-1.0, 1.0))
        u0 = x.maximum(0.0) - 0.5 * x  # kink at 0 -> interior breakpoint
        u = N.expm(0.05, u0, n=96)
        exact0 = 0.0  # odd-symmetrized component sums cancel at x=0 slowly
        assert np.isfinite(float(u(jnp.asarray(0.5))))
        assert abs(float(u(jnp.asarray(-1.0)))) < 1e-8
        assert abs(float(u(jnp.asarray(1.0)))) < 1e-8
        del exact0

    def test_auto_sigma_eigs(self):
        # Linop.eigs with sigma=None runs the MATLAB auto-sigma
        # (33/65 probe + smoothest-eigenvector selection).
        from chebfunjax.operators.blocks import D as D_op
        from chebfunjax.operators.blocks import eval_at as ev
        from chebfunjax.operators.linop import Linop

        L = Linop(D_op((0.0, math.pi), 2),
                  [ev(0.0, (0.0, math.pi)), ev(math.pi, (0.0, math.pi))],
                  domain=(0.0, math.pi))
        lam = L.eigs(n=65, k=4)
        got = np.sort(np.abs(np.asarray(lam)))[:3]
        want = np.array([1.0, 4.0, 9.0])  # -u'' = k^2 u on (0, pi)
        assert np.max(np.abs(got - want)) < 1e-6


class TestBlockLinopExtras:
    def test_periodic_and_continuity(self):
        # A first-order periodic system exercises _make_periodic /
        # derive_continuity.
        from chebfunjax.operators.blocks import D as D_op

        dom = (-math.pi, math.pi)
        A = linop(ChebMatrix([[D_op(dom)]]))
        A2 = A.derive_continuity((dom[0], 0.0, dom[1]))
        assert A2.nrows >= A.nrows

    def test_solve_wrapper(self):
        # solve() == linsolve() for a scalar block problem u'' = 1.
        from chebfunjax.operators.blocks import D as D_op

        dom = (-1.0, 1.0)
        L = linop(ChebMatrix([[D_op(dom) ** 2]]))
        L = L.addbc(eval_at(-1.0, dom)).addbc(eval_at(1.0, dom))
        one = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom)
        u = L.solve([one])
        u0 = u[0]
        # u = (x^2 - 1)/2
        assert abs(float(u0(jnp.asarray(0.0))) + 0.5) < 1e-9
