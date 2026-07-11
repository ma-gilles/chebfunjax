"""Port of MATLAB Chebfun tests/chebop/test_null.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_null.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="operator null space not implemented")


class TestChebopNull:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
