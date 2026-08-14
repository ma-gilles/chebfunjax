"""Port of MATLAB Chebfun tests/chebfun2/test_qr.m (Fable 5).

[Q, R, E] = qr(f): Q orthonormal quasimatrix in y, R quasimatrix in x
with R evaluated at the pivot x-locations E upper triangular, and
f = Q * R.

Provenance
----------
MATLAB source : tests/chebfun2/test_qr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.quadrature import chebpts, chebweights

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _check(dom):
    xa, xb, ya, yb = dom
    f = Chebfun2.from_function(
        lambda x, y: 1.0 / (10 + x ** 2 + y ** 2), domain=dom)
    Q, R, E = f.qr()
    k = len(f.approx.cols)

    xs = 0.5 * (xb - xa) * np.asarray(chebpts(100)) + 0.5 * (xa + xb)
    ys = 0.5 * (yb - ya) * np.asarray(chebpts(100)) + 0.5 * (ya + yb)
    xr = (2 * xs - (xa + xb)) / (xb - xa)
    yr = (2 * ys - (ya + yb)) / (yb - ya)
    Qv = np.asarray(Q(jnp.asarray(yr)))
    Rv = np.asarray(R(jnp.asarray(xr)))
    XX, YY = np.meshgrid(xs, ys)
    F = np.asarray(f(jnp.asarray(XX), jnp.asarray(YY)))

    # pass(1, 4): f == Q * R.
    assert np.max(np.abs(F - Qv @ Rv.T)) < np.sqrt(TOL)

    # pass(2, 5): Q' * Q == eye (continuous inner product on [ya, yb]).
    w = np.asarray(chebweights(100, kind=2)) * (yb - ya) / 2
    QtQ = (Qv * w[:, None]).T @ Qv
    assert np.max(np.abs(QtQ - np.eye(k))) < TOL

    # pass(3, 6): R evaluated at the pivot x-locations is triangular.
    Er = (2 * np.asarray(E) - (xa + xb)) / (xb - xa)
    M = np.asarray(R(jnp.asarray(Er))).T   # M[j, i] = R_j(E_i)
    assert np.max(np.abs(np.tril(M, -1))) < TOL


class TestChebfun2Qr:
    def test_default_domain(self):
        _check((-1.0, 1.0, -1.0, 1.0))

    def test_rectangle(self):
        _check((-2.1, 4.3, -1.0, 0.3))
