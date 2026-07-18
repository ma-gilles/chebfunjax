"""Port of MATLAB Chebfun tests/chebfun2/test_optimization.m (Fable 5).

Global optimization over the full 24-function battery via
``Chebfun2.minandmax2``.

Provenance
----------
MATLAB source : tests/chebfun2/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

# tol = 1000 * chebfun2eps (chebfun2eps default = eps = 2^-52)
TOL = 1000 * 2.220446049250313e-16

BATTERY = [
    lambda x, y: jnp.cos(np.pi * x * y),
    lambda x, y: jnp.cos(2 * np.pi * x * y),
    lambda x, y: jnp.cos(3 * np.pi * x * y),
    lambda x, y: jnp.cos(4 * np.pi * x * y),
    lambda x, y: jnp.cos(5 * np.pi * x * y),
    lambda x, y: jnp.cos(6 * np.pi * x * y),
    lambda x, y: jnp.cos(7 * np.pi * x * y),
    lambda x, y: jnp.sin(np.pi * x * y),
    lambda x, y: jnp.cos(0 * np.pi * (x - y) ** 2),
    lambda x, y: jnp.cos(np.pi * (x - y) ** 2),
    lambda x, y: jnp.cos(2 * np.pi * (x - y) ** 2),
    lambda x, y: jnp.exp(jnp.sin(4 * np.pi / (1 + x))
                         * jnp.sin(4 * np.pi / (1 + y))),
    lambda x, y: jnp.log(1 + x * y),
    lambda x, y: jnp.cos(2 * np.pi * x * jnp.sin(np.pi * y))
    + jnp.cos(2 * np.pi * y * jnp.sin(np.pi * x)),
    lambda x, y: (1 - x * y) / (1 + x ** 2 + y ** 2),
    lambda x, y: jnp.cos(np.pi * x * y ** 2) * jnp.cos(np.pi * y * x ** 2),
    lambda x, y: jnp.cos(2 * np.pi * x * y ** 2)
    * jnp.cos(2 * np.pi * y * x ** 2),
    lambda x, y: jnp.cos(3 * np.pi * x * y ** 2)
    * jnp.cos(3 * np.pi * y * x ** 2),
    lambda x, y: (x - y) / (2 - x ** 2 + y ** 2)
    + (y - x) / (2 - y ** 2 + x ** 2),
    lambda x, y: jnp.exp(-y * x ** 2) + jnp.exp(-x * y ** 2),
    lambda x, y: jnp.exp((1 - x ** 2) / (1 + y ** 2))
    + jnp.exp((1 - y ** 2) / (1 + x ** 2)),
    lambda x, y: 10.0 ** (-x * y),
    lambda x, y: 10.0 ** (-10 * x * y),
    lambda x, y: jnp.sin(x + y),
]

MAXI = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.exp(1), np.log(2), 2, 1, 1, 1, 1,
        2 / 3, 2, 2 * np.exp(1), 1, 1, 1]
MINI = [-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, np.exp(-1), 0, -2, 0,
        -0.132504231754118, -0.449023014530046, -0.805912853597402, 0,
        2 * np.exp(-1), 2, 1e-1, 1e-10, 0]


class TestChebfun2Optimization:
    def test_all_matlab_assertions(self):
        for jj, fn in enumerate(BATTERY):
            g = Chebfun2.from_function(fn, domain=(0.0, 1.0, 0.0, 1.0))
            Y, _X = g.minandmax2()
            err = (abs(float(Y[0]) - MINI[jj])
                   + abs(float(Y[1]) - MAXI[jj]))
            assert err < TOL, f"battery[{jj + 1}] err={err:.3e}"
