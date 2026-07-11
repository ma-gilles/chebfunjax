"""Port of MATLAB Chebfun tests/diskfun/test_sum2.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_sum2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-10


class TestDiskfunSum2:
    def test_area(self):
        one = Diskfun.from_function(lambda t, r: jnp.ones_like(r))
        assert abs(float(one.sum()) - np.pi) < 100 * TOL

    def test_gaussian_mass(self):
        # int_disk e^{-r^2} = pi (1 - e^{-1})
        f = Diskfun.from_function(lambda t, r: jnp.exp(-r ** 2))
        exact = np.pi * (1 - np.exp(-1))
        assert abs(float(f.sum()) - exact) < 100 * TOL
