"""Port of MATLAB Chebfun tests/chebfun3v/test_minandmax3est.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_minandmax3est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1000 * EPS


class TestChebfun3vMinandmax3est:
    def test_empty(self):
        f = Chebfun3v()
        assert f.minandmax3est().size == 0

    def test_one_component(self):
        f = Chebfun3v.from_functions(lambda x, y, z: x,
                                     domain=(-2, 4, -1, 1, -1, 1))
        mM = f.minandmax3est()
        assert mM.shape[0] == 2
        assert float(jnp.linalg.norm(mM - jnp.asarray([-2.0, 4.0]))) < TOL

    def test_three_components(self):
        f = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z,
                                     domain=(-2, 4, 3, 17, -1, 42))
        mM = f.minandmax3est()
        assert mM.shape[0] == 6
        expected = jnp.asarray([-2.0, 4.0, 3.0, 17.0, -1.0, 42.0])
        assert float(jnp.linalg.norm(mM - expected)) < TOL
