"""Port of MATLAB Chebfun tests/chebfun3v/test_minandmax3est.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_minandmax3est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun3v: 'minandmax3est' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun3vMinandmax3est:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
