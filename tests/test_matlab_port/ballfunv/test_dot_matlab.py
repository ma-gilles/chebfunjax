"""Port of MATLAB Chebfun tests/ballfunv/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_dot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfunv import Ballfunv

R0, L0, T0 = jnp.asarray(0.6), jnp.asarray(0.7), jnp.asarray(1.1)
X0 = float(R0 * jnp.cos(L0) * jnp.sin(T0))
Y0 = float(R0 * jnp.sin(L0) * jnp.sin(T0))
Z0 = float(R0 * jnp.cos(T0))


class TestBallfunvDot:
    def test_position_dot_itself(self):
        P = Ballfunv.from_functions(lambda x, y, z: x,
                                    lambda x, y, z: y,
                                    lambda x, y, z: z)
        d = P.dot(P)
        exact = X0 ** 2 + Y0 ** 2 + Z0 ** 2
        assert abs(float(d(R0, L0, T0)) - exact) < 1e-9
