"""Port of MATLAB Chebfun tests/chebfun2v/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vCurl:
    def test_scalar_curl(self):
        # curl(x^2, xy) = y; curl(-y, x) = 2
        F = Chebfun2v.from_functions(lambda x, y: x * x,
                                     lambda x, y: x * y)
        assert abs(float(F.curl()(X0, Y0)) - float(Y0)) < 1e-9
        G = Chebfun2v.from_functions(lambda x, y: -y, lambda x, y: x)
        assert abs(float(G.curl()(X0, Y0)) - 2.0) < 1e-10

    def test_curl_of_gradient_is_zero(self):
        F = Chebfun2v.from_functions(lambda x, y: y, lambda x, y: x)
        assert abs(float(F.curl()(X0, Y0))) < 1e-10
