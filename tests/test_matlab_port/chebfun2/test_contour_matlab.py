"""Port of MATLAB Chebfun tests/chebfun2/test_contour.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_contour.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot-output test; chebfunjax contour() delegates to matplotlib without the tested MATLAB return values")


class TestChebfun2Contour:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
