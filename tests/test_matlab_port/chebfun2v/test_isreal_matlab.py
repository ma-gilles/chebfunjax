"""Port of MATLAB Chebfun tests/chebfun2v/test_isreal.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


class TestChebfun2vIsreal:
    def test_real_field(self):
        # pass(1)
        f = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: y + 0 * x)
        assert f.isreal()

    def test_complex_field(self):
        # pass(2)
        f = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: 1j * (y + 0 * x))
        assert not f.isreal()

    def test_real_of_complex(self):
        # pass(3): real(chebfun2v(1i*x + y, y - x)) is real.
        f = Chebfun2v.from_functions(lambda x, y: 1j * x + y,
                                     lambda x, y: y - x)
        assert f.real().isreal()
