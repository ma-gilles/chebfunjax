"""Port of MATLAB Chebfun tests/chebop/test_deflate_bratu.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_deflate_bratu.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop has no deflation")


class TestChebopDeflateBratu:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
