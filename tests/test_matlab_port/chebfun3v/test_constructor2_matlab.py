"""Port of MATLAB Chebfun tests/chebfun3v/test_constructor2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_constructor2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 100 * EPS

# The MATLAB test exercises the 'vectorize' flag by contrasting a
# vectorized handle (x.*y.*z) with a scalar one (x*y*z).  In chebfunjax
# every handle is already jax-vectorized, so both branches use the same
# vectorized lambda and the fields must coincide.


class TestChebfun3vConstructor2:
    def test_two_components_default_domain(self):
        H1 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z)
        H2 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z)
        assert float((H1 - H2).norm()) < TOL

    def test_two_components_domain(self):
        dom = (-2, 3, -1, 0, -1, 1)
        H1 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z, domain=dom)
        H2 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z, domain=dom)
        assert float((H1 - H2).norm()) < TOL

    def test_three_components_default_domain(self):
        H1 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z)
        H2 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z)
        assert float((H1 - H2).norm()) < TOL

    def test_three_components_domain(self):
        dom = (-2, 3, -1, 0, -1, 1)
        H1 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z, domain=dom)
        H2 = Chebfun3v.from_functions(lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z,
                                      lambda x, y, z: x * y * z, domain=dom)
        assert float((H1 - H2).norm()) < TOL
