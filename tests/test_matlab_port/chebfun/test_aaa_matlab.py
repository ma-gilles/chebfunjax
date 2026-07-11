"""Port of MATLAB Chebfun tests/chebfun/test_aaa.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_aaa.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test drives chebfun-input aaa incl. Froissart cleanup options; chebfunjax aaa is array-based, covered by unit tests (NOT YET PORTED assertion-for-assertion)")


class TestChebfunAaa:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
