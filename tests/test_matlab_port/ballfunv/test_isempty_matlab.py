"""Port of MATLAB Chebfun tests/ballfunv/test_isempty.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no empty")


class TestBallfunvIsempty:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
