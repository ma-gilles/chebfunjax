"""Port of MATLAB Chebfun tests/spherefun/test_inherited.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_inherited.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="inherited separableApprox methods (flipud/trace/...) not implemented on Spherefun")


class TestSpherefunInherited:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
