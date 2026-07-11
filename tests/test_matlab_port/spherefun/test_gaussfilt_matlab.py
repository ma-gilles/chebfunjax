"""Port of MATLAB Chebfun tests/spherefun/test_gaussfilt.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_gaussfilt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no gaussfilt")


class TestSpherefunGaussfilt:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
