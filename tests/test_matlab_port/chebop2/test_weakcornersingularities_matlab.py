"""Port of MATLAB Chebfun tests/chebop2/test_weakcornersingularities.m (Opus 4.8).

Resolve a harmonic function with a weak corner singularity to high accuracy,
solved with the coefficient-space (ultraspherical) Chebop2 path.

Provenance
----------
MATLAB source : tests/chebop2/test_weakcornersingularities.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2

_EPS = float(np.finfo(np.float64).eps)


def _f(x, y):
    r2 = x ** 2 + y ** 2
    th = np.arctan(y / x)
    return r2 * (np.log(np.sqrt(r2)) * np.sin(2.0 * th) + th * np.cos(2.0 * th))


def _ev(u, x, y):
    return float(np.asarray(
        u(jnp.asarray([x], dtype=jnp.float64), jnp.asarray([y], dtype=jnp.float64))
    )[0])


class TestChebop2Weakcornersingularities:
    def test_all_matlab_assertions(self):
        tol = 1e6 * _EPS  # MATLAB 1e6 * cheb2Prefs.chebfun2eps.

        e = 1e-16  # step away from the corner (else f is NaN there).
        d = (e, 1.0 - e, e, 1.0 - e)
        N = Chebop2(lambda u: u.diff(0, 2) + u.diff(2, 0), domain=d)
        N.lbc = lambda y: _f(e, y)
        N.rbc = lambda y: _f(1.0 - e, y)
        N.dbc = lambda x: _f(x, e)
        N.ubc = lambda x: _f(x, 1.0 - e)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = N.solve(0.0)

        # pass(1) and pass(2): pointwise accuracy at interior points.
        assert abs(_f(np.pi / 6.0, np.pi / 12.0) - _ev(u, np.pi / 6.0, np.pi / 12.0)) < tol
        assert abs(_f(0.8, 0.23) - _ev(u, 0.8, 0.23)) < tol
