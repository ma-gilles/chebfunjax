"""Port of MATLAB Chebfun tests/chebfun2v/test_roots05.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots05.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'roots05' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vRoots05:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
