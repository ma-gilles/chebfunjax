"""Port of MATLAB Chebfun tests/chebfun3v/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

X0, Y0, Z0 = jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(0.5)

EPS = 2.220446049250313e-16
TOL = 10 * EPS


class TestChebfun3vCross:
    def test_xhat_cross_yhat(self):
        E1 = Chebfun3v.from_functions(lambda x, y, z: 1.0 + 0 * x,
                                      lambda x, y, z: 0 * x,
                                      lambda x, y, z: 0 * x)
        E2 = Chebfun3v.from_functions(lambda x, y, z: 0 * x,
                                      lambda x, y, z: 1.0 + 0 * x,
                                      lambda x, y, z: 0 * x)
        C = E1.cross(E2)
        comps = C.components
        assert abs(float(comps[0](X0, Y0, Z0))) < 1e-10
        assert abs(float(comps[1](X0, Y0, Z0))) < 1e-10
        assert abs(float(comps[2](X0, Y0, Z0)) - 1.0) < 1e-10

    def test_definition(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.exp(z))
        G = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        f1, f2, f3 = F.components
        g1, g2, g3 = G.components
        crossF = Chebfun3v([f2 * g3 - f3 * g2,
                            f3 * g1 - f1 * g3,
                            f1 * g2 - f2 * g1])
        assert float((F.cross(G) - crossF).norm()) < TOL

    def test_definition_second(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: x * y)
        G = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: x + y)
        f1, f2, f3 = F.components
        g1, g2, g3 = G.components
        crossF = Chebfun3v([f2 * g3 - f3 * g2,
                            f3 * g1 - f1 * g3,
                            f1 * g2 - f2 * g1])
        assert float((F.cross(G) - crossF).norm()) < TOL
