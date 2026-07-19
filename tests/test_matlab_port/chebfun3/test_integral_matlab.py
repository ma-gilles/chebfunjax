"""Port of MATLAB Chebfun tests/chebfun3/test_integral.m (Fable 5).

FIXED (Fable 5): Chebfun3.integral (line integral over a parametric
curve) added in the audit.  The curve is supplied as a callable
``t -> (x(t), y(t), z(t))`` together with its parameter interval.

Provenance
----------
MATLAB source : tests/chebfun3/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e3 * EPS


class TestChebfun3Integral:
    def test_all_matlab_assertions(self):
        # pass(1)
        f = Chebfun3.from_function(lambda x, y, z: 2 * x * y + 3 * z)
        exact = 38 * np.sqrt(5)
        val = f.integral(lambda t: (2 * t, 5 * t, 4 * t), domain=(0, 1))
        assert abs(val - exact) / exact < TOL

        # pass(2)
        f = Chebfun3.from_function(lambda x, y, z: x ** 2 * z)
        exact = 23 * np.sqrt(30) / 12
        val = f.integral(lambda t: (-t, 6 + 2 * t, 2 + 5 * t),
                         domain=(0, 1))
        assert abs(val - exact) / exact < TOL

        # pass(3): helix over an enlarged z-domain.
        dom = (-1.0, 1.0, -1.0, 1.0, 0.0, 12 * np.pi)
        f = Chebfun3.from_function(lambda x, y, z: x * y * z, domain=dom)
        exact = -3 * np.sqrt(10) * np.pi
        val = f.integral(
            lambda t: (jnp.cos(t), jnp.sin(t), 3 * t),
            domain=(0, 4 * np.pi))
        assert abs(val - exact) / abs(exact) < TOL
