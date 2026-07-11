"""Port of MATLAB Chebfun tests/chebfun2v/test_plotting.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'plotting' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vPlotting:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
