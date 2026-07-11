"""Port of MATLAB Chebfun tests/ballfunv/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="ballfunv feature 'laplacian' not implemented (MATLAB-specific accessor or missing op)")


class TestBallfunvLaplacian:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
