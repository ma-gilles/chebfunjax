"""Port of MATLAB Chebfun tests/chebfun/test_trigpade.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_trigpade.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no trigpade")


class TestChebfunTrigpade:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
