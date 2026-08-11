"""Port of MATLAB Chebfun tests/chebfun2/test_cdr.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun2.cdr()`` now exists.
MATLAB's quasimatrices ``C`` and ``R`` become lists of 1D Chebfuns, so
MATLAB's ``C(y,:)`` is ``np.stack([c(y) for c in C], axis=-1)``.  The
single-output MATLAB form ``d = cdr(f)`` (the pivot vector) is
``np.diag(D)`` here.

Provenance
----------
MATLAB source : tests/chebfun2/test_cdr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


def _quasi_eval(cols, t):
    """MATLAB's C(t, :): evaluate every column at t, as an (len(t), r) matrix."""
    return np.stack([np.asarray(c(jnp.asarray(t))) for c in cols], axis=-1)


class TestChebfun2Cdr:
    def test_cdr_reconstructs_on_unit_square(self):
        # pass(1): f(xx, yy) == C(x,:) * D * R(x,:).'
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        x = np.linspace(-1.0, 1.0, 100)
        xx, yy = np.meshgrid(x, x)
        C, D, R = f.cdr()
        approx = _quasi_eval(C, x) @ np.asarray(D) @ _quasi_eval(R, x).T
        exact = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy)))
        assert float(np.linalg.norm(exact - approx)) < TOL

    def test_cdr_reconstructs_on_rectangle(self):
        # pass(2): the same on [-3 4 -1 10], where C lives on the
        # y-interval and R on the x-interval.
        dom = (-3.0, 4.0, -1.0, 10.0)
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=dom)
        x = np.linspace(dom[0], dom[1], 100)
        y = np.linspace(dom[2], dom[3], 100)
        xx, yy = np.meshgrid(x, y)
        C, D, R = f.cdr()
        approx = _quasi_eval(C, y) @ np.asarray(D) @ _quasi_eval(R, x).T
        exact = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy)))
        assert float(np.linalg.norm(exact - approx)) < TOL

    def test_cdr_pivot_vector(self):
        # pass(3): the diagonal of D is the pivot vector.
        dom = (-3.0, 4.0, -1.0, 10.0)
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=dom)
        _, D, _ = f.cdr()
        d = np.asarray(f.approx.pivots)
        assert float(np.linalg.norm(np.diag(np.asarray(D)) - d)) < TOL
