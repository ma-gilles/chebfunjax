"""Port of MATLAB Chebfun tests/chebfun3/test_get.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB get() property interface has no counterpart")


class TestChebfun3Get:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
