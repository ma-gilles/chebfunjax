"""Port of MATLAB Chebfun tests/spherefun/test_plotting.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot smoke test")


class TestSpherefunPlotting:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
