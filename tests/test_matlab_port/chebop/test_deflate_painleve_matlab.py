"""Port of MATLAB Chebfun tests/chebop/test_deflate_painleve.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_deflate_painleve.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop has no deflation")


class TestChebopDeflatePainleve:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
