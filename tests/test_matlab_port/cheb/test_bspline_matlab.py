"""Port of MATLAB Chebfun tests/cheb/test_bspline.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_bspline.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no cheb.bspline")


class TestChebBspline:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
