"""Original 22-statement empty command test; source asserts no exception only.

Provenance
----------
MATLAB source : tests/chebfun3/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Emptyobjects:
    def test_all_commands_tolerate_empty(self):
        f = Chebfun3.empty()
        f + f
        2 * f
        f * 2
        f ** 2
        2 ** f
        f ** f
        f.sqrt()
        f.sum()
        f.sum2()
        f.sum3()
        f.integral()
        f.diff()
        f.sin()
        f.cos()
        f.sinh()
        f ** f + f
        f.tucker()
        f.mean()
        f.max3()
        f.norm()
        f.minandmax3()
        f.permute()
