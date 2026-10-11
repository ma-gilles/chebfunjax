"""Supplemental analytic position-field norm check.

This is not a port of a native test file: tests/diskfunv/test_norm.m does
not exist at Chebfun commit 7574c77. The independent integral oracle below
checks the library operation documented in @diskfunv/norm.m at that pin.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfunv import Diskfunv


class TestDiskfunvNorm:
    def test_norm_of_position_field(self):
        # ||(x,y)||^2 = int_disk r^2 = pi/2
        P = Diskfunv.from_functions(lambda t, r: r * jnp.cos(t),
                                    lambda t, r: r * jnp.sin(t))
        n = float(P.norm())
        assert abs(n - np.sqrt(np.pi / 2)) < 1e-8
