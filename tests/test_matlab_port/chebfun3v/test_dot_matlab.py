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

EPS = 2.220446049250313e-16
TOL = 50 * EPS


class TestChebfun3vDot:
    def test_position_dot(self):
        P = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        d = P.dot(P)
        exact = 0.3 ** 2 + 0.4 ** 2 + 0.5 ** 2
        assert abs(float(d(X0, Y0, Z0)) - exact) < 1e-9

    def test_dot_equals_ctranspose_times(self):
        # dot(F, G) == F' * G  (MATLAB definition).
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.exp(z))
        G = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        dotF1 = F.dot(G)
        dotF2 = F.ctranspose() @ G
        assert float((dotF1 - dotF2).norm()) < TOL

    def test_dot_equals_ctranspose_times_second(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: x * y)
        G = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        dotF1 = F.dot(G)
        dotF2 = F.ctranspose() @ G
        assert float((dotF1 - dotF2).norm()) < TOL
