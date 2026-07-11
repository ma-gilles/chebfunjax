"""Port of MATLAB Chebfun tests/chebmatrix/test_matrixOutput.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_matrixOutput.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebmatrix (block operator/chebfun container) class")


class TestChebmatrixMatrixoutput:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
