"""Port of MATLAB Chebfun tests/domain/test_merge.m (Fable 5).

Provenance
----------
MATLAB source : tests/domain/test_merge.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Domain has no merge (breakpoint union is internal to arithmetic)")


class TestDomainMerge:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
