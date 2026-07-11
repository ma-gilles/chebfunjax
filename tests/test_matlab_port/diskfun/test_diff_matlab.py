"""Port of MATLAB Chebfun tests/diskfun/test_diff.m (Fable 5).

Cartesian partials of polynomial functions on the disk are exact:
d/dx (x y) = y, d/dy (x y) = x, etc.

Provenance
----------
MATLAB source : tests/diskfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-9


class TestDiskfunDiff:
    def test_cartesian_partials_of_xy(self):
        # FIXED in the Fable 5 audit: the reconstruct's radial lstsq
        # fit carried ~1e-11 noise that sent the constructor's GE down
        # a noise-pivot path; now a well-conditioned Chebyshev-Gauss
        # solve.
        f = Diskfun.from_function(
            lambda t, r: r ** 2 * jnp.cos(t) * jnp.sin(t))
        fx = f.diffx()
        fy = f.diffy()
        t0 = jnp.asarray(0.6)
        r0 = jnp.asarray(0.7)
        y = float(r0 * jnp.sin(t0))
        x = float(r0 * jnp.cos(t0))
        assert abs(float(fx(t0, r0)) - y) < TOL
        assert abs(float(fy(t0, r0)) - x) < TOL

    def test_higher_partial(self):
        # passes: x^2 -> 2x (m=2,cos + m=0 dx transition happens to
        # cancel correctly here), then 2x -> 2 (m=1, working class)
        # d^2/dx^2 (x^2) = 2
        f = Diskfun.from_function(lambda t, r: (r * jnp.cos(t)) ** 2)
        fxx = f.diffx().diffx()
        assert abs(float(fxx(jnp.asarray(0.3), jnp.asarray(0.5))) - 2.0) \
            < 10 * TOL
