"""Port of MATLAB Chebfun tests/chebfun2/test_guide.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_guide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="guide script exercises max2/mean/cumsum/composition (absent)")


class TestChebfun2Guide:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
