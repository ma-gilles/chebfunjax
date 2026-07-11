"""Port of MATLAB Chebfun tests/chebfun2v/test_divergence.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_divergence.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vDivergence:
    def test_divergence_of_polynomial_field(self):
        # div(x^2, xy) = 2x + x = 3x
        F = Chebfun2v.from_functions(lambda x, y: x * x,
                                     lambda x, y: x * y)
        d = F.div()
        assert abs(float(d(X0, Y0)) - 3 * float(X0)) < 1e-9

    def test_rotation_divergence_free(self):
        F = Chebfun2v.from_functions(lambda x, y: -y, lambda x, y: x)
        assert abs(float(F.div()(X0, Y0))) < 1e-10
