"""Port of MATLAB Chebfun tests/ballfun/test_sample.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sample.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 2.220446049250313e-16


class TestBallfunSample:
    def test_all_matlab_assertions(self):
        # Example 1
        f = Ballfun.from_function(lambda x, y, z: x)
        X = np.array(f.sample())
        Y = np.zeros((2, 3, 3))
        Y[1, :, 1] = [-1.0, 0.5, 0.5]
        assert np.linalg.norm((X - Y).ravel()) < TOL  # pass(1)

        # Example 2
        f = Ballfun.from_function(lambda x, y, z: z)
        X = np.array(f.sample(3, 1, 4))
        r = np.array([0.0, np.sqrt(0.5), 1.0])
        cosT = np.array([1.0, 0.5, -0.5, -1.0])
        Y = np.outer(r, cosT).reshape(3, 1, 4)
        assert np.linalg.norm((X - Y).ravel()) < TOL  # pass(2)

        # Example 3: values -> ballfun round trip
        g = Ballfun.from_values(np.array(f.sample(4, 4, 4)))
        assert (g - f).norm() < TOL  # pass(3)
