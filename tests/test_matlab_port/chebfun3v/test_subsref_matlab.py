"""Port of MATLAB Chebfun tests/chebfun3v/test_subsref.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_subsref.m
Chebfun commit: 7574c77

Notes
-----
The MATLAB test also composes with three CHEBFUN2 objects (producing a
CHEBFUN2V), with 1D CHEBFUN quasimatrices, and with a SPHEREFUNV.  Those
composition targets are on the CHEBFUN2V / CHEBFUN / SPHEREFUNV side and are
not part of the chebfun3v surface, so only the component-access and
CHEBFUN3/CHEBFUN3V compositions are ported here.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1000 * EPS


class TestChebfun3vSubsref:
    def test_component_access(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x * y * z))
        F = Chebfun3v([f, f, f])
        # F(1) returns the first component; its core matches f's core.
        assert float(jnp.linalg.norm(
            jnp.asarray(F[0].core).ravel() - jnp.asarray(f.core).ravel())) < TOL

    def test_compose_two_component_target(self):
        # G(f1, f2, f3) with the identity CHEBFUN3s returns G.
        f1 = Chebfun3.from_function(lambda x, y, z: x)
        f2 = Chebfun3.from_function(lambda x, y, z: y)
        f3 = Chebfun3.from_function(lambda x, y, z: z)
        G = Chebfun3v.from_functions(lambda x, y, z: x + y,
                                     lambda x, y, z: y)
        H = Chebfun3v([f1, f2, f3]).compose(G)
        assert float((H - G).norm()) < TOL

    def test_compose_three_component_target(self):
        f1 = Chebfun3.from_function(lambda x, y, z: x)
        f2 = Chebfun3.from_function(lambda x, y, z: y)
        f3 = Chebfun3.from_function(lambda x, y, z: z)
        G = Chebfun3v.from_functions(lambda x, y, z: x + y,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        H = Chebfun3v([f1, f2, f3]).compose(G)
        assert float((H - G).norm()) < TOL

    def test_compose_chebfun3v_with_chebfun3v(self):
        dom = (-2, 2, -2, 2, 0, 2)
        G = Chebfun3v.from_functions(lambda x, y, z: x + y + z,
                                     lambda x, y, z: x - y, domain=dom)
        F = Chebfun3v.from_functions(lambda x, y, z: 2 * x,
                                     lambda x, y, z: x + y,
                                     lambda x, y, z: z + 1)
        H = F.compose(G)
        H_true = Chebfun3v.from_functions(lambda x, y, z: 3 * x + y + z + 1,
                                          lambda x, y, z: x - y)
        assert float((H - H_true).norm()) < TOL
