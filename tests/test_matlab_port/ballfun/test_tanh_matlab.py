"""Port of MATLAB Chebfun tests/ballfun/test_tanh.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_tanh.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no tanh composition")


class TestBallfunTanh:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
