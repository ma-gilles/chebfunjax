"""Port of MATLAB Chebfun tests/chebfun3/test_root.m (Fable 5).

root(f, g, h): one common root of three chebfun3 objects.

Provenance
----------
MATLAB source : tests/chebfun3/test_root.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)


class TestChebfun3Root:
    def test_curve_intersection(self):
        # pass(1-3): y = x^2, z = x^3 intersected with an oscillatory h.
        f = Chebfun3.from_function(lambda x, y, z: y - x ** 2)
        g = Chebfun3.from_function(lambda x, y, z: z - x ** 3)
        h = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(jnp.exp(x * jnp.sin(-2 + y + z))))
        r = Chebfun3.root(f, g, h)
        pt = [jnp.asarray(float(v)) for v in np.asarray(r)]
        tol = 10 * EPS
        assert abs(float(f(*pt))) < 1e3 * tol
        assert abs(float(g(*pt))) < 1e3 * tol
        assert abs(float(h(*pt))) < 1e3 * tol

    def test_separable_roots(self):
        # pass(4-6): x^2 = y^2 = z^2 = 1e-2 — the root lands on one of
        # the eight (+-0.1, +-0.1, +-0.1) points.
        c = 1e-2
        f = Chebfun3.from_function(lambda x, y, z: x ** 2 - c)
        g = Chebfun3.from_function(lambda x, y, z: y ** 2 - c)
        h = Chebfun3.from_function(lambda x, y, z: z ** 2 - c)
        r = np.asarray(Chebfun3.root(f, g, h))
        assert np.max(np.abs(np.abs(r) - 0.1)) < 1e-8
