"""Port of MATLAB Chebfun tests/chebfun2v/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'size' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vSize:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
