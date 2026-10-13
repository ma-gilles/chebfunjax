"""Port of MATLAB Chebfun tests/chebfun3/test_iszero.m (Fable 5).

FIXED (Fable 5): Original scalar, empty and callback constructions are preserved.

Provenance
----------
MATLAB source : tests/chebfun3/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import chebfun3


class TestChebfun3Iszero:
    def test_all_matlab_assertions(self):
        # chebfun3(0): the zero function.
        f = chebfun3(0)
        assert f.iszero()

        # chebfun3([]): the empty object is treated as zero.
        assert chebfun3([]).iszero()

        # chebfun3(2): a nonzero constant.
        f = chebfun3(2)
        assert not f.iszero()

        f = chebfun3(lambda x, y, z: x + y - z)
        assert not f.iszero()
