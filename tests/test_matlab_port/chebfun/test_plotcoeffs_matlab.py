"""Port of MATLAB Chebfun tests/chebfun/test_plotcoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_plotcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot smoke test")


class TestChebfunPlotcoeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
