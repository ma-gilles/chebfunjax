"""Port of MATLAB Chebfun tests/chebfun/test_comet.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_comet.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot animation; no counterpart")


class TestChebfunComet:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
