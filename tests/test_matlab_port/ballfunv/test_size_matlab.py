"""Port of MATLAB Chebfun tests/ballfunv/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

jax.config.update("jax_enable_x64", True)


class TestBallfunvSize:
    def test_all_matlab_assertions(self):
        # Example 1
        Vx = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        Vy = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        Vz = Ballfun.from_function(lambda x, y, z: x)
        S = np.array([[1, 1, 1], [1, 1, 1], [2, 3, 3]])
        V = Ballfunv(Vx, Vy, Vz)
        assert np.array_equal(S, V.size)  # pass(1)
