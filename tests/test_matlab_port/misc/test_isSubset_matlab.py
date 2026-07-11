"""Port of MATLAB Chebfun tests/misc/test_isSubset.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_isSubset.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no isSubset domain utility")


class TestMiscIssubset:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
