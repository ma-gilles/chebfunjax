"""Port of MATLAB Chebfun tests/ballfun/test_mean3.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_mean3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no mean3")


class TestBallfunMean3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
