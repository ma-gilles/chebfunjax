"""Port of MATLAB Chebfun tests/chebfun3v/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun3v: 'laplacian' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun3vLaplacian:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
