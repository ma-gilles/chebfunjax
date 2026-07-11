"""Port of MATLAB Chebfun tests/ballfunv/test_HelmholtzDecomposition.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_HelmholtzDecomposition.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="ballfunv feature 'HelmholtzDecomposition' not implemented (MATLAB-specific accessor or missing op)")


class TestBallfunvHelmholtzdecomposition:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
