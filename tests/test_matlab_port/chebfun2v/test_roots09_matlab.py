"""Port of MATLAB Chebfun tests/chebfun2v/test_roots09.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots09.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'roots09' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vRoots09:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
