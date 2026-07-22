"""Core-suite mirrors for chebop periodic/piecewise eigs and interior jumps.

Direct functional checks (closed-form or self-consistent) for the paths the
MATLAB-port suite covers with golden references:

* periodic eigenvalues by Fourier collocation (``eigs`` with ``bc='periodic'``);
* a generalized eigenproblem with a discontinuous mass coefficient
  (block-diagonal pencil + continuity rows);
* an interior jump condition in a general ``.bc`` (piecewise solve);
* a scalar ODE whose ``sign(x)`` coefficient induces an interior breakpoint.

Provenance
----------
Mirrors of MATLAB @chebop/@linop eigs.m and mldivide.m jump handling;
Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from chebfunjax.chebfun1d.chebfun import chebfun, jump  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestPeriodicEigs:
    def test_laplacian_eigenvalues(self):
        # -u'' = lambda u, periodic on [0, 2pi]: eigenvalues 0,1,1,4,4,...
        L = Chebop(lambda u: -u.diff(2), (0.0, 2 * np.pi))
        L.bc = "periodic"
        D = np.sort(np.real(np.asarray(L.eigs(k=5))))
        assert np.max(np.abs(D - np.array([0.0, 1.0, 1.0, 4.0, 4.0]))) < 1e-9


class TestPiecewiseGeneralizedEigs:
    def test_discontinuous_mass_smallest_magnitude(self):
        ep = 0.25
        x = chebfun(lambda t: t, domain=(-1.0, 1.0))
        F = (x.abs() < ep) * (1.0 / (2 * ep))
        L = Chebop(lambda xx, u: u.diff(2), (-1.0, 1.0))
        L.lbc = 0.0
        L.rbc = 0.0
        M = Chebop(lambda xx, u: F * u, (-1.0, 1.0))
        _, ec = L.eigs_generalized(M, k=3)
        ec = np.sort(np.real(np.asarray(ec)))
        # Smallest three (nearest zero) from the reference #1074 solution.
        ref = np.sort(1.0e2 * np.array([
            -0.841876384196023, -0.247291299790162, -0.023950791540263]))
        assert ec.shape == (3,)
        assert np.max(np.abs(ec - ref)) < 5e-9


class TestInteriorJump:
    def test_nonhomogeneous_jump(self):
        # V'' = 0, V(-1)=-1, V(1)=1, jump(V,0)=1, V' continuous at 0.
        N = Chebop(lambda s, V: V.diff(2), (-1.0, 1.0))
        N.lbc = lambda V: V + 1
        N.rbc = lambda V: V - 1
        N.bc = lambda s, V: [jump(V, 0.0) - 1, jump(V.diff(), 0.0)]
        y = N.solve(0.0, n=8)
        assert abs((float(y(0.0, "right")) - float(y(0.0, "left"))) - 1) < 1e-10
        assert abs(float(y.diff()(0.0, "right"))
                   - float(y.diff()(0.0, "left"))) < 1e-10
        assert abs(float(y(-1.0)) + 1) < 1e-10
        assert abs(float(y(1.0)) - 1) < 1e-10


class TestSignCoefficientBreakpoint:
    def test_induced_breakpoint_solve(self):
        # diff(u,2) + sign(x) sin(u) on [-1, 0.5, 1]; sign(x) induces a break
        # at 0 that the piecewise solver must add to converge.
        N = Chebop(lambda x, u: u.diff(2) + x.sign() * u.sin(),
                   (-1.0, 0.5, 1.0))
        N.lbc = lambda u: u - 2
        N.rbc = lambda u: u - 2
        u = N.solve(0.0)
        assert float(N(u).norm()) < 5e-10
        # Boundary conditions satisfied.
        assert abs(float(u(-1.0)) - 2) < 1e-9
        assert abs(float(u(1.0)) - 2) < 1e-9
