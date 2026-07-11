"""Port of MATLAB Chebfun tests/chebfun2/test_roots.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2.roots returns sampled zero-contour point arrays, not chebfun curves; the MATLAB assertions integrate/measure chebfun-valued contours")


class TestChebfun2Roots:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
