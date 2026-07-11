"""Port of MATLAB Chebfun tests/chebfun/test_aaatrig.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_aaatrig.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no trig-AAA (aaatrig)")


class TestChebfunAaatrig:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
