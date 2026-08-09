"""Port of MATLAB Chebfun tests/spherefun/test_roots.m (Fable 5).

FIXED (Fable 5): Spherefun.roots (zero contours on the sphere) via the
chebfun2 zero-curve engine on the (lambda, theta) view of the sphere,
mapped to Cartesian (x, y, z) point arrays.

Provenance
----------
MATLAB source : tests/spherefun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e5 * float(np.finfo(np.float64).eps)


def _xyz(lam, th):
    return (jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th),
            jnp.cos(th))


class TestSpherefunRoots:
    def test_simple_contours(self):
        # pass(1): z has a zero contour (the equator).
        f = Spherefun.from_function(lambda lam, th: _xyz(lam, th)[2])
        assert len(f.roots()) > 0

        # pass(2): 2 + z > 0 everywhere -> no contour.
        f = Spherefun.from_function(lambda lam, th: 2 + _xyz(lam, th)[2])
        assert len(f.roots()) == 0

        # pass(3): z=0 contour lies on the equator (z-coordinate ~ 0).
        f = Spherefun.from_function(lambda lam, th: _xyz(lam, th)[2])
        r = f.roots()
        assert float(np.max(np.abs(r[0][:, 2]))) < TOL

        # pass(4): x=0 contour (x-coordinate ~ 0).
        f = Spherefun.from_function(lambda lam, th: _xyz(lam, th)[0])
        r = f.roots()
        assert float(np.max(np.abs(r[0][:, 0]))) < TOL

        # pass(5): y=0 contour (y-coordinate ~ 0).
        f = Spherefun.from_function(lambda lam, th: _xyz(lam, th)[1])
        r = f.roots()
        assert float(np.max(np.abs(r[0][:, 1]))) < TOL

    # Formerly a strict xfail (worst contour 1.4e-3 vs the 1e-3 MATLAB
    # tolerance): the 2026-08 SeparableApprox global-tolerance slice
    # chop improved the chebfun2-view re-approximation at the great-
    # circle crossings and the residual now clears the MATLAB bound.
    def test_sinh_residual(self):
        def fn(lam, th):
            x, y, z = _xyz(lam, th)
            return 2 * jnp.sinh(5 * x * y * z)

        f = Spherefun.from_function(fn)
        r = f.roots()
        assert len(r) > 0
        for contour in r:
            x, y, z = contour[:, 0], contour[:, 1], contour[:, 2]
            lam = np.arctan2(y, x)
            th = np.arccos(np.clip(z, -1.0, 1.0))
            fvals = np.asarray(f(jnp.asarray(lam), jnp.asarray(th)))
            assert float(np.max(np.abs(fvals))) < 1e-3
