"""Port of MATLAB Chebfun tests/chebfun3/test_sum.m (Fable 5).

MATLAB sum(f) integrates over one variable and returns a chebfun2.
chebfunjax Chebfun3.sum(dim) exists; assertions compare values against
the analytic 2-D integral on a lattice.

Provenance
----------
MATLAB source : tests/chebfun3/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e4 * EPS


class TestChebfun3Sum:
    def test_sum_over_z_of_xyz(self):
        # int_{-1}^{1} x*y*z^2 dz = (2/3) x*y
        f = Chebfun3.from_function(lambda x, y, z: x * y * z ** 2)
        s = f.sum(dim=3)
        xs = np.linspace(-0.9, 0.9, 8)
        for xv in xs:
            for yv in xs[::3]:
                got = float(s(jnp.asarray(xv), jnp.asarray(yv)))
                assert abs(got - (2.0 / 3.0) * xv * yv) < 10 * TOL

    def test_sum_over_x_of_odd_is_zero(self):
        f = Chebfun3.from_function(lambda x, y, z: x * jnp.cos(y * z))
        s = f.sum(dim=1)
        got = float(s(jnp.asarray(0.4), jnp.asarray(0.3)))
        assert abs(got) < 10 * TOL
