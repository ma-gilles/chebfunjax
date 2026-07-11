"""Port of MATLAB Chebfun tests/chebfun/test_end.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_end.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB end-indexing has no counterpart")


class TestChebfunEnd:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
