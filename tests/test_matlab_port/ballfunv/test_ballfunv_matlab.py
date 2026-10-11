"""Supplemental ball-vector construction control.

These are supplemental Python checks, not translated native test assertions.
The formerly cited native test file does not exist at Chebfun7574c77.
The corresponding library operation is documented in @ballfunv/ballfunv.m;
the analytic formulas and diagnostic bounds below are supplemental oracles.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfunv import Ballfunv

R0, L0, T0 = jnp.asarray(0.6), jnp.asarray(0.7), jnp.asarray(1.1)


class TestBallfunvBallfunv:
    def test_construction_components(self):
        F = Ballfunv.from_functions(lambda x, y, z: x,
                                    lambda x, y, z: y,
                                    lambda x, y, z: z)
        assert len(F.components) == 3
        x0 = float(R0 * jnp.cos(L0) * jnp.sin(T0))
        assert abs(float(F.components[0](R0, L0, T0)) - x0) < 1e-9
