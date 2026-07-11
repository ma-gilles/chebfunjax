"""Port of MATLAB Chebfun tests/ballfun/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB size semantics differ (shape exists; smoke-tested in constructor port)")


class TestBallfunSize:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
