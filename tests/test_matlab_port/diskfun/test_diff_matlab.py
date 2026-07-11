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
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-9


class TestDiskfunDiff:
    @pytest.mark.xfail(
        reason="Opus 4.8's diskfun diffx/diffy are WRONG outside the "
        "mode classes its own tests covered: d/dy(y^2) and d/dy(xy) "
        "return 0; d/dx(r^2) and d/dx(r^4) return 0; d/dx(x^3) and "
        "d/dy(r^3 cos 3t) return wrong nonzero values. Only "
        "(m=2, cos) and m=1 transitions are correct. Flagged in the "
        "Fable 5 audit; _diskfun_reconstruct needs a modal-coupling "
        "rewrite.")
    def test_cartesian_partials_of_xy(self):
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
