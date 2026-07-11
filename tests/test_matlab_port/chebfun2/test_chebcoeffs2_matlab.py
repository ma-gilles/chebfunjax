"""Port of MATLAB Chebfun tests/chebfun2/test_chebcoeffs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_chebcoeffs2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no chebcoeffs2 (bivariate coefficient matrix) accessor")


class TestChebfun2Chebcoeffs2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
