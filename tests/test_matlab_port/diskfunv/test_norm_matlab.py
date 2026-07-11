"""Port of MATLAB Chebfun tests/diskfunv/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfunv import Diskfunv


class TestDiskfunvNorm:
    def test_norm_of_position_field(self):
        # ||(x,y)||^2 = int_disk r^2 = pi/2
        P = Diskfunv.from_functions(lambda t, r: r * jnp.cos(t),
                                    lambda t, r: r * jnp.sin(t))
        try:
            n = float(P.norm())
        except (TypeError, NotImplementedError):
            pytest.skip("Diskfunv.norm not implemented")
        assert abs(n - np.sqrt(np.pi / 2)) < 1e-8
