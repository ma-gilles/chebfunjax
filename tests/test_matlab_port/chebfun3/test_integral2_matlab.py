"""Port of MATLAB Chebfun tests/chebfun3/test_integral2.m (Fable 5).

FIXED (Fable 5): Chebfun3.integral2 (surface integral over a parametric
surface) added in the audit.  The surface is supplied as a callable
``(u, v) -> (x(u,v), y(u,v), z(u,v))`` together with its parameter
rectangle.

Provenance
----------
MATLAB source : tests/chebfun3/test_integral2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e7 * EPS


class TestChebfun3Integral2:
    def test_all_matlab_assertions(self):
        # Surface S(u,v) = (u cos v, u sin v, v); f = sqrt(1 + x^2 + y^2).
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.sqrt(1 + x ** 2 + y ** 2),
            domain=(-4.0, 4.0, -4.0, 4.0, 0.0, 2 * np.pi))
        I = f.integral2(
            lambda u, v: (u * jnp.cos(v), u * jnp.sin(v), v),
            domain=(0.0, 4.0, 0.0, 2 * np.pi))
        exact = 152 * np.pi / 3
        assert abs(I - exact) < TOL
