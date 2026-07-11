"""Port of MATLAB Chebfun tests/cheb/test_gallery2.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_gallery2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no cheb.gallery2 (chebfun2 gallery)")


class TestChebGallery2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
