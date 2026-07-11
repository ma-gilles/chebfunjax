"""Port of MATLAB Chebfun tests/ballfunv/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfunv import Ballfunv

R0, L0, T0 = jnp.asarray(0.6), jnp.asarray(0.7), jnp.asarray(1.1)
X0 = float(R0 * jnp.cos(L0) * jnp.sin(T0))
Y0 = float(R0 * jnp.sin(L0) * jnp.sin(T0))
Z0 = float(R0 * jnp.cos(T0))


class TestBallfunvCross:
    def test_xhat_cross_yhat(self):
        E1 = Ballfunv.from_functions(lambda x, y, z: 1 + 0 * x,
                                     lambda x, y, z: 0 * x,
                                     lambda x, y, z: 0 * x)
        E2 = Ballfunv.from_functions(lambda x, y, z: 0 * x,
                                     lambda x, y, z: 1 + 0 * x,
                                     lambda x, y, z: 0 * x)
        C = E1.cross(E2)
        vals = [float(c(R0, L0, T0)) for c in C.components]
        assert abs(vals[0]) < 1e-10 and abs(vals[1]) < 1e-10
        assert abs(vals[2] - 1.0) < 1e-10
