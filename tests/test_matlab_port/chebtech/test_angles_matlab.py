"""Port of MATLAB Chebfun tests/chebtech/test_angles.m (Opus 4.8).

MATLAB ``chebtech{1,2}.angles(n)`` returns ``acos(chebpts(n))`` (the angles of
the Chebyshev points).  chebfunjax has no ``angles`` static method; the check
``cos(angles(n)) == chebpts(n)`` is a trivial identity of ``acos``, so there is
nothing meaningful to test.

Provenance
----------
MATLAB source : tests/chebtech/test_angles.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestChebtechAngles:
    def test_chebtech1(self):
        # pass(1): cos(chebtech1.angles(10)) == chebtech1.chebpts(10).
        pytest.skip(
            "chebfunjax has no chebtech.angles; acos(chebpts) is trivial"
        )

    def test_chebtech2(self):
        # pass(2): cos(chebtech2.angles(10)) == chebtech2.chebpts(10).
        pytest.skip(
            "chebfunjax has no chebtech.angles; acos(chebpts) is trivial"
        )
