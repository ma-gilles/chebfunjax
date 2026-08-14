"""Port of MATLAB Chebfun tests/chebfun2/test_lu.m (Fable 5).

[L, U] = lu(f): f = L * U with L unit-valued at the pivot
y-locations and U upper triangular at the pivot x-locations.

Provenance
----------
MATLAB source : tests/chebfun2/test_lu.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _check(dom):
    xa, xb, ya, yb = dom
    f = Chebfun2.from_function(
        lambda x, y: jnp.cos(x * y), domain=dom)
    L, U, piv = f.lu()

    xs = 0.5 * (xb - xa) * np.asarray(chebpts(100)) + 0.5 * (xa + xb)
    ys = 0.5 * (yb - ya) * np.asarray(chebpts(100)) + 0.5 * (ya + yb)
    xr = (2 * xs - (xa + xb)) / (xb - xa)
    yr = (2 * ys - (ya + yb)) / (yb - ya)
    Lv = np.asarray(L(jnp.asarray(yr)))
    Uv = np.asarray(U(jnp.asarray(xr)))
    XX, YY = np.meshgrid(xs, ys)
    F = np.asarray(f(jnp.asarray(XX), jnp.asarray(YY)))

    # pass(1, 4): f == L * U.
    assert np.max(np.abs(F - Lv @ Uv.T)) < 1e2 * TOL

    # pass(2, 5): diag(L at pivot y-locations) == 1.
    yr_p = (2 * np.asarray([xy[1] for xy in piv]) - (ya + yb)) / (yb - ya)
    Ldiag = np.diag(np.asarray(L(jnp.asarray(yr_p))))
    assert np.max(np.abs(Ldiag - 1.0)) < TOL

    # pass(3, 6): U at the pivot x-locations is upper triangular.
    xr_p = (2 * np.asarray([xy[0] for xy in piv]) - (xa + xb)) / (xb - xa)
    M = np.asarray(U(jnp.asarray(xr_p))).T
    assert np.max(np.abs(np.tril(M, -1))) < np.sqrt(TOL)


class TestChebfun2Lu:
    def test_default_domain(self):
        _check((-1.0, 1.0, -1.0, 1.0))

    def test_rectangle(self):
        _check((-2.1, 4.3, -1.0, 2.7))
