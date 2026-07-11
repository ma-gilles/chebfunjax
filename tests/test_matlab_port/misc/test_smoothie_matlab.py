"""Port of MATLAB Chebfun tests/misc/test_smoothie.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_smoothie.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks chebfun-valued smoothie; chebfunjax returns grid samples (NOT YET PORTED assertion-for-assertion)")


class TestMiscSmoothie:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
