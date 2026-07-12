"""Port of MATLAB Chebfun tests/chebfun3/test_restrict.m (Fable 5).

FIXED: Chebfun3.restrict added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1e3 * np.finfo(float).eps
DOM = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)


def _ff(x, y, z):
    return jnp.exp(x / 2 + y) + jnp.cos(x + z ** 2)


class TestChebfun3Restrict:
    def test_all_matlab_assertions(self):
        f = Chebfun3.from_function(_ff, domain=DOM)
        d = DOM
        # pass(1): restrict to a point
        val = f.restrict((-2.5, -2.5, 0, 0, 2.5, 2.5))
        exact = float(_ff(jnp.asarray(-2.5), jnp.asarray(0.0),
                          jnp.asarray(2.5)))
        assert abs(val - exact) < TOL * 20

        # pass(2): vertical line
        f1 = f.restrict((d[0], d[1], 0, 0, 2.5, 2.5))
        xs = jnp.asarray(np.linspace(d[0], d[1], 15))
        assert float(jnp.max(jnp.abs(
            f1(xs) - _ff(xs, 0 * xs, 0 * xs + 2.5)))) < TOL * 20

        # pass(3): horizontal line
        f2 = f.restrict((d[0], d[0], d[2], d[3], d[4], d[4]))
        ys = jnp.asarray(np.linspace(d[2], d[3], 15))
        assert float(jnp.max(jnp.abs(
            f2(ys) - _ff(0 * ys + d[0], ys, 0 * ys + d[4])))) \
            < TOL * 20

        # pass(4): oblique (z) line
        f3 = f.restrict((d[0], d[0], d[2], d[2], d[4], d[5]))
        zs = jnp.asarray(np.linspace(d[4], d[5], 15))
        assert float(jnp.max(jnp.abs(
            f3(zs) - _ff(0 * zs + d[0], 0 * zs + d[2], zs)))) \
            < TOL * 20

        # pass(5): plane x = d[0]
        f4 = f.restrict((d[0], d[0], d[2], d[3], d[4], d[5]))
        yy, zz = jnp.meshgrid(ys, zs, indexing="ij")
        assert float(jnp.max(jnp.abs(
            f4(yy, zz) - _ff(0 * yy + d[0], yy, zz)))) < TOL * 20
