"""Port of MATLAB Chebfun tests/diskfun/test_optimization.m (Fable 5).

Global optimization over the disk via ``minandmax2``.

Provenance
----------
MATLAB source : tests/diskfun/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

# tol = 20000 * chebfun2eps (chebfun2eps default = eps = 2^-52)
TOL = 20000 * 2.220446049250313e-16


def _cart(fn):
    """diskfun(@(x,y) fn) with x = r cos(theta), y = r sin(theta)."""
    return Diskfun.from_function(
        lambda t, r: fn(r * jnp.cos(t), r * jnp.sin(t)))


class TestDiskfunOptimization:
    def test_all_matlab_assertions(self):
        battery = [
            lambda x, y: jnp.cos(np.pi * x),
            lambda x, y: jnp.cos(np.pi * y),
            lambda x, y: jnp.cos(np.pi * x) * jnp.cos(np.pi * y),
            lambda x, y: jnp.cos(2 * np.pi * x) * jnp.cos(2 * np.pi * y),
            lambda x, y: jnp.exp(-10 * ((x - 1 / np.sqrt(2)) ** 2 + y ** 2)),
        ]
        maxi = [1.0, 1.0, 1.0, 1.0, 1.0]
        mini = [-1.0, -1.0, -1.0, -1.0, 0.0]
        for jj, fn in enumerate(battery):
            g = _cart(fn)
            Y, _X = g.minandmax2()
            err = abs(float(Y[0]) - mini[jj]) + abs(float(Y[1]) - maxi[jj])
            assert err < TOL
