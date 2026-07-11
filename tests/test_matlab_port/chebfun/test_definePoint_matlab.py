"""Port of MATLAB Chebfun tests/chebfun/test_definePoint.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_definePoint.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB subsasgn point assignment has no counterpart")


class TestChebfunDefinepoint:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
