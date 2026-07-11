"""Port of MATLAB Chebfun tests/cheb/test_galleryball.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_galleryball.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no cheb.galleryball")


class TestChebGalleryball:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
