"""Port of MATLAB Chebfun tests/spherefun/test_rotate.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_rotate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no rotate")


class TestSpherefunRotate:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
