"""Port of MATLAB Chebfun tests/functionalBlock/test_isNotMultOrDiff.m (Fable 5).

Provenance
----------
MATLAB source : tests/functionalBlock/test_isNotMultOrDiff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax functional blocks are internal; covered via chebop BC handling tests")


class TestFunctionalblockIsnotmultordiff:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
