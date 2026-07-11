"""Port of MATLAB Chebfun tests/diskfunv/test_divergence.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_divergence.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfunv import Diskfunv

T0, R0 = jnp.asarray(0.6), jnp.asarray(0.7)
X0 = float(R0 * jnp.cos(T0))
Y0 = float(R0 * jnp.sin(T0))


class TestDiskfunvDivergence:
    def test_rotation_field_divergence_free(self):
        F = Diskfunv.from_functions(lambda t, r: -r * jnp.sin(t),
                                    lambda t, r: r * jnp.cos(t))
        assert abs(float(F.div()(T0, R0))) < 1e-10

    @pytest.mark.xfail(
        reason="inherits the diskfun _diskfun_reconstruct modal bug: "
        "div(x^2, xy) returns 0 instead of 3x (see the diskfun diff "
        "port; Fable 5 audit)")
    def test_polynomial_field(self):
        F = Diskfunv.from_functions(
            lambda t, r: (r * jnp.cos(t)) ** 2,
            lambda t, r: r ** 2 * jnp.cos(t) * jnp.sin(t))
        assert abs(float(F.div()(T0, R0)) - 3 * X0) < 1e-8
