"""Port of MATLAB Chebfun tests/ballfunv/test_PTdecomposition.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_PTdecomposition.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="ballfunv feature 'PTdecomposition' not implemented (MATLAB-specific accessor or missing op)")


class TestBallfunvPtdecomposition:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
