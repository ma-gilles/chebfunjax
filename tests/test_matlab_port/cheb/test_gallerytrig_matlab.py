"""Port of MATLAB Chebfun tests/cheb/test_gallerytrig.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_gallerytrig.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.utils.gallerytrig import gallerytrig, list_gallerytrig

FAST = sorted(set(list_gallerytrig()) - {"tsunami"})


class TestChebGallerytrig:
    @pytest.mark.parametrize("name", FAST)
    def test_does_not_crash(self, name):
        assert gallerytrig(name) is not None
