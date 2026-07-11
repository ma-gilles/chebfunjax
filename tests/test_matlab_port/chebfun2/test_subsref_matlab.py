"""Port of MATLAB Chebfun tests/chebfun2/test_subsref.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB subsref indexing semantics have no Python counterpart")


class TestChebfun2Subsref:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
