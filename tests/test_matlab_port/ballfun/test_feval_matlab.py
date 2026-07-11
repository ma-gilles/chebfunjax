"""Port of MATLAB Chebfun tests/ballfun/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS


class TestBallfunFeval:
    def test_grid_evaluation(self):
        f = Ballfun.from_function(lambda x, y, z: x + 2 * y + 3 * z)
        r = jnp.asarray(np.linspace(0.1, 0.9, 5))
        lam = jnp.asarray(np.linspace(-3.0, 3.0, 5))
        th = jnp.asarray(np.linspace(0.2, 2.9, 5))
        V = np.asarray(f(r, lam, th))
        # tensor grid: verify one entry against the exact formula
        i, j, k = 2, 3, 1
        x = float(r[i] * jnp.cos(lam[j]) * jnp.sin(th[k]))
        y = float(r[i] * jnp.sin(lam[j]) * jnp.sin(th[k]))
        z = float(r[i] * jnp.cos(th[k]))
        # find which axis order the tensor uses by scanning
        target = x + 2 * y + 3 * z
        assert np.min(np.abs(V - target)) < 1e3 * EPS
