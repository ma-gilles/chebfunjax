"""Port of MATLAB Chebfun tests/chebfun/test_subspace.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_subspace.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no array-valued (multi-column) chebfun")


class TestChebfunSubspace:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
