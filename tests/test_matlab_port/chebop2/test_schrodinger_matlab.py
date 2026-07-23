"""Port of MATLAB Chebfun tests/chebop2/test_schrodinger.m (Opus 4.8).

The constant-coefficient Schrodinger equation ``i*u_t + u_xx - V*u = 0`` has a
complex operator coefficient.  Solved with the coefficient-space (ultraspherical)
Chebop2 path, which carries the complex coefficient through the separable-rank
expansion and the (complex) reduced solve.

MATLAB compares ``norm(u - exact)`` (chebfun2 L2 norm); we use a fine-grid L2
quadrature of the complex-valued difference at the MATLAB tolerances.

Provenance
----------
MATLAB source : tests/chebop2/test_schrodinger.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2

_EPS = float(np.finfo(np.float64).eps)


def _l2err(u, exact, dom, ng=150):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, ng)
    ys = np.linspace(ya, yb, ng)
    X, Y = np.meshgrid(xs, ys)
    e = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()))) \
        - exact(X.ravel(), Y.ravel())
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    return float(np.sqrt(np.sum(np.abs(e) ** 2) * dx * dy))


def _solve(exact, V, dom):
    N = Chebop2(lambda u: 1j * u.diff(1, 0) + u.diff(0, 2) - V * u, domain=dom)
    N.lbc = lambda t: exact(dom[0], t)
    N.rbc = lambda t: exact(dom[1], t)
    N.dbc = lambda x: exact(x, dom[2])
    return N.solve(0.0)


class TestChebop2Schrodinger:
    def test_all_matlab_assertions(self):
        tol = 1000.0 * _EPS  # MATLAB 1000 * cheb2Prefs.chebfun2eps.

        # pass(1): w > V, travelling plane wave on [-10,10]x[0,1].
        V, w = 1.0, 2.0
        k = np.sqrt(w - V)
        d = (-10.0, 10.0, 0.0, 1.0)
        exact = lambda x, t: np.exp(1j * (k * x - w * t))
        assert _l2err(_solve(exact, V, d), exact, d) < 10.0 * tol

        # pass(2): superposition of two counter-propagating waves.
        exact = lambda x, t: (np.exp(1j * (k * x - w * t))
                              + np.exp(1j * (-k * x - w * t)))
        assert _l2err(_solve(exact, V, d), exact, d) < 10.0 * tol

        # pass(3): w < V (evanescent, complex wave number) on [-1,1]x[0,1].
        V, w = 2.0, 1.0
        k = np.sqrt(complex(w - V))
        d = (-1.0, 1.0, 0.0, 1.0)
        exact = lambda x, t: np.exp(1j * (k * x - w * t))
        assert _l2err(_solve(exact, V, d), exact, d) < tol
