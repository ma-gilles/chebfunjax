"""Port of MATLAB Chebfun tests/ballfunv/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Ballfunv has no arithmetic")


class TestBallfunvTimes:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
