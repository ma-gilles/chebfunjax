"""Port of MATLAB Chebfun tests/chebfun/test_constructor_splitting.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_splitting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="covered by tests/test_chebfun1d SplittingOn tests (chebfunjax splitting=True)")


class TestChebfunConstructorSplitting:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
