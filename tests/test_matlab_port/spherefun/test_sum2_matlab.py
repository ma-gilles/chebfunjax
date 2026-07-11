"""Port of MATLAB Chebfun tests/spherefun/test_sum2.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_sum2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-10


class TestSpherefunSum2:
    def test_surface_area(self):
        one = Spherefun.from_function(
            lambda lam, th: jnp.ones_like(th))
        assert abs(float(one.sum()) - 4 * np.pi) < 100 * TOL

    def test_odd_harmonic_integrates_to_zero(self):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        assert abs(float(f.sum())) < 100 * TOL
