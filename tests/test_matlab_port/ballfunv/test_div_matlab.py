"""Analytic polynomial ball-vector divergence controls.

These are supplemental Python checks, not translated native test assertions.
The formerly cited native test file does not exist at Chebfun7574c77.
The corresponding library operation is documented in @ballfunv/divergence.m;
the analytic formulas and diagnostic bounds below are supplemental oracles.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfunv import Ballfunv

R0, L0, T0 = jnp.asarray(0.6), jnp.asarray(0.7), jnp.asarray(1.1)


class TestBallfunvDiv:
    def test_divergence_free_rotation(self):
        F = Ballfunv.from_functions(lambda x, y, z: -y,
                                    lambda x, y, z: x,
                                    lambda x, y, z: 0 * x)
        assert abs(float(F.div()(R0, L0, T0))) < 1e-10

    def test_linear_field(self):
        # div(x, 2y, 3z) = 6
        F = Ballfunv.from_functions(lambda x, y, z: x,
                                    lambda x, y, z: 2 * y,
                                    lambda x, y, z: 3 * z)
        assert abs(float(F.div()(R0, L0, T0)) - 6.0) < 1e-8
