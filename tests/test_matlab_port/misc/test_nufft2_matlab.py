"""Port of MATLAB Chebfun tests/misc/test_nufft2.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_nufft2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no 2-D nufft2")


class TestMiscNufft2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
