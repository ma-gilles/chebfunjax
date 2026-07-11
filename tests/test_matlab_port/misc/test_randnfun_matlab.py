"""Port of MATLAB Chebfun tests/misc/test_randnfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks chebfun-valued randnfun statistics; chebfunjax randnfun covered by tests/test_utils/test_randnfun.py")


class TestMiscRandnfun:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
