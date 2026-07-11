"""Port of MATLAB Chebfun tests/chebfun3/test_sum2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_sum2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e4 * EPS


class TestChebfun3Sum2:
    def test_sum2_over_xy(self):
        # int int x^2*y^2*g(z) dx dy = (4/9) g(z)
        f = Chebfun3.from_function(lambda x, y, z: x ** 2 * y ** 2
                                   * jnp.cos(z))
        s = f.sum2(dims=(1, 2))
        for zv in np.linspace(-0.9, 0.9, 7):
            got = float(s(jnp.asarray(zv)))
            assert abs(got - (4.0 / 9.0) * np.cos(zv)) < 10 * TOL
