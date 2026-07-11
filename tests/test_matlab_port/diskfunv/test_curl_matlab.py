"""Port of MATLAB Chebfun tests/diskfunv/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfunv import Diskfunv

T0, R0 = jnp.asarray(0.6), jnp.asarray(0.7)
Y0 = float(R0 * jnp.sin(T0))


class TestDiskfunvCurl:
    def test_rotation_field_curl_two(self):
        F = Diskfunv.from_functions(lambda t, r: -r * jnp.sin(t),
                                    lambda t, r: r * jnp.cos(t))
        assert abs(float(F.curl()(T0, R0)) - 2.0) < 1e-9

    def test_polynomial_field(self):
        # FIXED with the diskfun radial-fit repair (Fable 5 audit).
        F = Diskfunv.from_functions(
            lambda t, r: (r * jnp.cos(t)) ** 2,
            lambda t, r: r ** 2 * jnp.cos(t) * jnp.sin(t))
        assert abs(float(F.curl()(T0, R0)) - Y0) < 1e-8
