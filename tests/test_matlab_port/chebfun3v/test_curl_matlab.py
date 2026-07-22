"""Port of MATLAB Chebfun tests/chebfun3v/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 50 * EPS


class TestChebfun3vCurl:
    def test_definition(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: x * y)
        f1, f2, f3 = F.components
        # curl = [f3_y - f2_z; f1_z - f3_x; f2_x - f1_y]
        curlF = Chebfun3v([f3.diff(2) - f2.diff(3),
                           f1.diff(3) - f3.diff(1),
                           f2.diff(1) - f1.diff(2)])
        assert float((curlF - F.curl()).norm()) < TOL

    def test_curl_of_gradient_is_zero(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x + y ** 2) + jnp.sin(y) + y)
        F = Chebfun3v.gradient(f)
        assert float(F.curl().norm()) < TOL
