"""Port of MATLAB Chebfun tests/chebfun3/test_isreal.m (Fable 5).

FIXED (Fable 5): Chebfun3.isreal added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Isreal:
    def test_all_matlab_assertions(self):
        f = Chebfun3.from_function(lambda x, y, z: x + y - z)
        assert f.isreal()

        f = Chebfun3.from_function(lambda x, y, z: 1j * x + y - z)
        assert not f.isreal()

        f = Chebfun3.from_function(lambda x, y, z: 1j * x + y - z).real()
        assert f.isreal()
