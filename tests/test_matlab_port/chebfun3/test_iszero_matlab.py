"""Port of MATLAB Chebfun tests/chebfun3/test_iszero.m (Fable 5).

FIXED (Fable 5): Chebfun3.iszero added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Iszero:
    def test_all_matlab_assertions(self):
        # chebfun3(0): the zero function.
        f = Chebfun3.from_function(lambda x, y, z: 0.0 * x)
        assert f.iszero()

        # chebfun3([]): the empty object is treated as zero.
        assert Chebfun3.empty().iszero()

        # chebfun3(2): a nonzero constant.
        f = Chebfun3.from_function(lambda x, y, z: 2.0 + 0.0 * x)
        assert not f.iszero()

        f = Chebfun3.from_function(lambda x, y, z: x + y - z)
        assert not f.iszero()
