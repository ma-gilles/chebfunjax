"""Port of MATLAB Chebfun tests/chebfun3v/test_isreal.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


class TestChebfun3vIsreal:
    def test_real_field(self):
        f = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y - z)
        assert f.isreal()

    def test_complex_field(self):
        f = Chebfun3v.from_functions(lambda x, y, z: 1j * x,
                                     lambda x, y, z: y - z)
        assert not f.isreal()

    def test_real_of_complex(self):
        f = Chebfun3v.from_functions(lambda x, y, z: 1j * x,
                                     lambda x, y, z: y - z).real()
        assert f.isreal()
