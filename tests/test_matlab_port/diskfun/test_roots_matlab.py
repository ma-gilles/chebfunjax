"""Port of MATLAB Chebfun tests/diskfun/test_roots.m (Fable 5).

FIXED (Fable 5): Diskfun.roots (zero contours + common zeros) via the
chebfun2 zero-curve engine on the polar (theta, r) view of the disk.

Provenance
----------
MATLAB source : tests/diskfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e5 * float(np.finfo(np.float64).eps)
_X = np.sort(np.cos(np.pi * np.arange(257) / 256))  # chebpts(257)
_XJ = jnp.asarray(_X)


class TestDiskfunRoots:
    def test_all_matlab_assertions(self):
        # pass(1): x + y has zero contours.
        f = Diskfun.from_function(
            lambda t, r: r * jnp.cos(t) + r * jnp.sin(t))
        assert len(f.roots()) > 0

        # pass(2): 2 + x^2 > 0 everywhere -> no contours.
        f = Diskfun.from_function(
            lambda t, r: 2 + (r * jnp.cos(t)) ** 2)
        assert len(f.roots()) == 0

        # pass(3): r^2 - 1/4 (polar) -> circle 0.5*exp(1i*pi*x).
        f = Diskfun.from_function(lambda t, r: r ** 2 - 0.5 ** 2)
        r = f.roots()
        assert len(r) == 1
        exact = 0.5 * np.exp(1j * _X * np.pi)
        assert float(np.max(np.abs(np.asarray(r[0](_XJ)) - exact))) < TOL

        # pass(4,5): cos(5r) -> two circles at r = pi/10 and 3*pi/10.
        f = Diskfun.from_function(lambda t, r: jnp.cos(5 * r))
        r = f.roots()
        assert len(r) == 2
        for radius in (np.pi / 10.0, 3 * np.pi / 10.0):
            exact = radius * np.exp(1j * _X * np.pi)
            best = min(float(np.max(np.abs(np.asarray(c(_XJ)) - exact)))
                       for c in r)
            assert best < TOL

        # pass(6): common zeros of (x^2+y^2-1/4) and x -> (0, +-1/2).
        f = Diskfun.from_function(
            lambda t, r: (r * jnp.cos(t)) ** 2 + (r * jnp.sin(t)) ** 2 - 0.25)
        gx = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
        pts = f.roots(gx)
        got = np.array(sorted(pts.tolist(), key=lambda p: p[1]))
        assert np.max(np.abs(got - np.array([[0, -0.5], [0, 0.5]]))) < TOL
