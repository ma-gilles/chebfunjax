"""Port of MATLAB Chebfun tests/chebfun2/test_isPeriodicTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_isPeriodicTech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="isPeriodicTech() exists on Chebfun2 but only the Chebyshev tech is reachable: the constructor has no 'trig' option to build a periodic Chebfun2 to test it against"
)


class TestChebfun2Isperiodictech:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
