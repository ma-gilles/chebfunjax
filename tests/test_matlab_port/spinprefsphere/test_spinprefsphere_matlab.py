"""Port of MATLAB Chebfun tests/spinprefsphere/test_spinprefsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinprefsphere/test_spinprefsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no spinsphere")


class TestSpinprefsphereSpinprefsphere:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
