"""Port of MATLAB Chebfun tests/diskfunv/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfunv import Diskfunv

T0, R0 = jnp.asarray(0.6), jnp.asarray(0.7)


class TestDiskfunvConstructor:
    def test_components(self):
        F = Diskfunv.from_functions(lambda t, r: r * jnp.cos(t),
                                    lambda t, r: r * jnp.sin(t))
        assert abs(float(F.components[0](T0, R0))
                   - float(R0 * jnp.cos(T0))) < 1e-10
        assert abs(float(F.components[1](T0, R0))
                   - float(R0 * jnp.sin(T0))) < 1e-10
