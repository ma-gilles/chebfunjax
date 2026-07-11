"""Port of MATLAB Chebfun tests/ballfunv/test_iszero.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no iszero")


class TestBallfunvIszero:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
