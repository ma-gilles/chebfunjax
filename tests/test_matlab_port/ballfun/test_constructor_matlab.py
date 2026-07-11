"""Port of MATLAB Chebfun tests/ballfun/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, Z0, val


class TestBallfunConstructor:
    def test_smooth_accuracy(self):
        f = Ballfun.from_function(
            lambda x, y, z: jnp.exp(x) * jnp.sin(y + z))
        exact = float(np.exp(X0) * np.sin(Y0 + Z0))
        assert abs(val(f) - exact) < 1e4 * EPS

    def test_roundtrip_coeffs(self):
        f = Ballfun.from_function(lambda x, y, z: x * y + z)
        g = Ballfun.from_coeffs(f.coeffs)
        assert abs(val(f) - val(g)) < 100 * EPS
