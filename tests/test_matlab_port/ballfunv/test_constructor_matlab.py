"""Port of MATLAB Chebfun tests/ballfunv/test_constructor.m (Fable 5).

MATLAB's ``ballfun(ones(n,n,n))`` values construction maps to
``Ballfun.from_values``; the ``[vx; vy; vz]`` bracket syntax maps to
``Ballfun.vertcat(vx, vy, vz)``.

Provenance
----------
MATLAB source : tests/ballfunv/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 2.220446049250313e-16


class TestBallfunvConstructor:
    def test_all_matlab_assertions(self):
        # Example 1: ballfun from a values cube, ballfunv from ballfuns
        n = 10
        f = Ballfun.from_values(np.ones((n, n, n)))
        g = Ballfunv(f, f, f)
        assert g is not None  # pass(1) = 1 in MATLAB

        # Example 2: ballfunv from function handles == from ballfuns
        v = Ballfunv.from_functions(
            lambda x, y, z: x * z,
            lambda x, y, z: y,
            lambda x, y, z: y * x,
        )
        vx = Ballfun.from_function(lambda x, y, z: x * z)
        vy = Ballfun.from_function(lambda x, y, z: y)
        vz = Ballfun.from_function(lambda x, y, z: y * x)
        w = Ballfunv(vx, vy, vz)
        assert (v - w).norm() < TOL  # pass(2)

        # Example 3: vertcat of three ballfuns makes a ballfunv
        v3 = Ballfun.vertcat(vx, vy, vz)
        assert (v3 - w).norm() < TOL  # pass(3)
