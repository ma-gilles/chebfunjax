"""Port of MATLAB Chebfun tests/chebfun/test_complex.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no complex(f, g) constructor (f + 1j*g covers it; complex arithmetic tested in imag/plus ports)")


class TestChebfunComplex:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
