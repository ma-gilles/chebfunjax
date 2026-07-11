"""Port of MATLAB Chebfun tests/chebfun/test_defineInterval.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_defineInterval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB subsasgn interval redefinition has no counterpart")


class TestChebfunDefineinterval:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
