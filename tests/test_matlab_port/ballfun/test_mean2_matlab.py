"""Port of MATLAB Chebfun tests/ballfun/test_mean2.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_mean2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no mean2")


class TestBallfunMean2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
