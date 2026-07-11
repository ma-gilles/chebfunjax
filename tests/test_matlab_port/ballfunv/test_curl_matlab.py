"""Port of MATLAB Chebfun tests/ballfunv/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfunv import Ballfunv

R0, L0, T0 = jnp.asarray(0.6), jnp.asarray(0.7), jnp.asarray(1.1)


class TestBallfunvCurl:
    def test_curl_of_rotation(self):
        F = Ballfunv.from_functions(lambda x, y, z: -y,
                                    lambda x, y, z: x,
                                    lambda x, y, z: 0 * x)
        c = F.curl()
        vals = [float(ci(R0, L0, T0)) for ci in c.components]
        assert abs(vals[0]) < 1e-9
        assert abs(vals[1]) < 1e-9
        assert abs(vals[2] - 2.0) < 1e-9

    def test_curl_of_gradient_is_zero(self):
        F = Ballfunv.from_functions(lambda x, y, z: y * z,
                                    lambda x, y, z: x * z,
                                    lambda x, y, z: x * y)
        c = F.curl()
        for ci in c.components:
            assert abs(float(ci(R0, L0, T0))) < 1e-9
