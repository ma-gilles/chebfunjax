"""Port of MATLAB Chebfun tests/chebfun3/test_minandmax3est.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_minandmax3est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no minandmax3est")


class TestChebfun3Minandmax3est:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
