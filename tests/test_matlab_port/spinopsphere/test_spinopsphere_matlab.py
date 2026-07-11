"""Port of MATLAB Chebfun tests/spinopsphere/test_spinopsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinopsphere/test_spinopsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no spinsphere")


class TestSpinopsphereSpinopsphere:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
