"""Port of MATLAB Chebfun tests/chebfun/test_get.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB get() property interface has no counterpart")


class TestChebfunGet:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
