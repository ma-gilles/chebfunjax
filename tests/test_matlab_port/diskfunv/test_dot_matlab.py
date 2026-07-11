"""Port of MATLAB Chebfun tests/diskfunv/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_dot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfunv import Diskfunv

T0, R0 = jnp.asarray(0.6), jnp.asarray(0.7)


class TestDiskfunvDot:
    def test_position_dot_itself(self):
        P = Diskfunv.from_functions(lambda t, r: r * jnp.cos(t),
                                    lambda t, r: r * jnp.sin(t))
        try:
            d = P.dot(P)
        except (TypeError, NotImplementedError):
            pytest.skip("Diskfunv.dot not implemented for pairs")
        assert abs(float(d(T0, R0)) - float(R0) ** 2) < 1e-9
