"""Port of MATLAB Chebfun tests/classicfun/test_isempty.m (Fable 5).

Provenance
----------
MATLAB source : tests/classicfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no empty Bndfun representation")


class TestClassicfunIsempty:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
