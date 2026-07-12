"""Port of MATLAB Chebfun tests/chebfun2/test_restriction.m (Fable 5).

FIXED: Chebfun2.restrict added in the Fable 5 audit (point, line,
chebfun-path restriction).

Provenance
----------
MATLAB source : tests/chebfun2/test_restriction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 1e5 * np.finfo(float).eps


class TestChebfun2Restriction:
    def test_all_matlab_assertions(self):
        def ff(x, y):
            return jnp.exp(-10 * (x ** 2 + y ** 2))

        g = Chebfun2.from_function(ff)
        # pass(1): restrict to a point
        assert abs(g.restrict((0, 0, 0, 0)) - 1.0) < TOL
        # pass(2): restrict to a line, evaluate at 0
        line = g.restrict((0, 0, -0.9, 0.1))
        assert abs(float(line(jnp.asarray(0.0))) - 1.0) < TOL
        # pass(3): restrict along a chebfun path t + 1e-18i
        path = cj.chebfun(lambda t: t + 1e-18 * 1j * t ** 0)
        rc = g.restrict(path)
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 21))
        assert float(jnp.max(jnp.abs(
            rc(xs) - jnp.exp(-10 * xs ** 2)))) < TOL
        # pass(4): restrict to the vertical line x = pi/6
        v = g.restrict((np.pi / 6, np.pi / 6, -1.0, 1.0))
        assert float(jnp.max(jnp.abs(
            v(xs) - jnp.exp(-10 * ((np.pi / 6) ** 2 + xs ** 2))))) \
            < TOL
