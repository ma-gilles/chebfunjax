"""Port of MATLAB Chebfun tests/misc/test_trigBary.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_trigBary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no trigBary (trigonometric barycentric interpolation)")


class TestMiscTrigbary:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
