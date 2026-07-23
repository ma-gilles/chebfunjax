"""Port of MATLAB Chebfun tests/chebop2/test_construction.m (Opus 4.8).

Checks the Chebop2 constructor and boundary-condition syntaxes on Laplace
problems whose solution is the constant 1, solved with the coefficient-space
(ultraspherical) Chebop2 path.

Ported: pass(1)-(4) (constant solutions on default and shifted domains, with
scalar / callable / chebfun-style Dirichlet data) and the "Neumann just runs"
smoke check.  The final block builds ``N.coeffs`` for a variable-coefficient
operator ``@(x,y,u) ... + x.*u`` (no assertion in MATLAB); that ``@(x,y,u)``
constructor syntax is a separate feature and is exercised in the
variable-coefficient ports.

Provenance
----------
MATLAB source : tests/chebop2/test_construction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, laplacian

_EPS = float(np.finfo(np.float64).eps)


def _maxerr_const(u, dom, npts=40):
    xs = np.linspace(dom[0], dom[1], npts)
    ys = np.linspace(dom[2], dom[3], npts)
    X, Y = np.meshgrid(xs, ys)
    uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    return float(np.max(np.abs(uv - 1.0)))


class TestChebop2Construction:
    def test_all_matlab_assertions(self):
        tol = 10.0 * _EPS  # MATLAB 10 * techPrefs.chebfuneps.

        # pass(1): laplacian, all edges = 1 -> u == 1.
        N = Chebop2(laplacian)
        N.lbc = 1.0
        N.rbc = 1.0
        N.ubc = 1.0
        N.dbc = 1.0
        assert _maxerr_const(N.solve(0.0), (-1.0, 1.0, -1.0, 1.0)) < tol

        # pass(2): laplacian written as diff(u,2,1)+diff(u,2,2).
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        N.lbc = 1.0
        N.rbc = 1.0
        N.ubc = 1.0
        N.dbc = 1.0
        assert _maxerr_const(N.solve(0.0), (-1.0, 1.0, -1.0, 1.0)) < tol

        # pass(3): on [0,1]^2.
        d = (0.0, 1.0, 0.0, 1.0)
        N = Chebop2(laplacian, domain=d)
        N.lbc = 1.0
        N.rbc = 1.0
        N.ubc = 1.0
        N.dbc = 1.0
        assert _maxerr_const(N.solve(0.0), d) < tol

        # pass(4): mixed BC syntax (callable, "chebfun-style" callable, scalars).
        N = Chebop2(laplacian)
        N.lbc = lambda x: 1.0 + 0.0 * x
        N.rbc = lambda x: 1.0 + 0.0 * x
        N.ubc = 1.0
        N.dbc = 1.0
        assert _maxerr_const(N.solve(0.0), (-1.0, 1.0, -1.0, 1.0)) < tol

        # Neumann conditions: just check the solve runs.
        N = Chebop2(laplacian)
        N.lbc = lambda x, u: u.diff(1)
        N.rbc = lambda x: 1.0 + 0.0 * x
        N.ubc = 1.0
        N.dbc = 1.0
        _ = N.solve(0.0)
