"""Port of MATLAB Chebfun tests/chebfun3v/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_dot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

X0, Y0, Z0 = jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(0.5)


class TestChebfun3vDot:
    def test_position_dot(self):
        P = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        d = P.dot(P)
        exact = 0.3 ** 2 + 0.4 ** 2 + 0.5 ** 2
        assert abs(float(d(X0, Y0, Z0)) - exact) < 1e-9
