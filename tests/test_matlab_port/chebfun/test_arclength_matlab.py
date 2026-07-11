"""Port of MATLAB Chebfun tests/chebfun/test_arclength.m (Fable 5).

FIXED: Chebfun.arclength added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_arclength.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunArclength:
    def test_line(self):
        f = cj.chebfun(lambda x: 2 * x, domain=(0.0, 1.0))
        assert abs(float(f.arclength()) - np.sqrt(5)) < 1e-12

    def test_unit_circle_half(self):
        # arc length of sqrt(1-x^2) on [-r, r] subtends 2 asin(r)
        r = 0.8
        f = cj.chebfun(lambda x: jnp.sqrt(1 - x * x), domain=(-r, r))
        assert abs(float(f.arclength()) - 2 * np.arcsin(r)) < 1e-8
