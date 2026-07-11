"""Port of MATLAB Chebfun tests/spherefun/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no spherefunv vertical concatenation")


class TestSpherefunVertcat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
