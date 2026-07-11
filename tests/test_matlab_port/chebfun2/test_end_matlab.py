"""Port of MATLAB Chebfun tests/chebfun2/test_end.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_end.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB end-indexing has no Python counterpart on Chebfun2")


class TestChebfun2End:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
