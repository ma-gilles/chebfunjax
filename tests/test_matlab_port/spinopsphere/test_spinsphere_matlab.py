"""Port of MATLAB Chebfun tests/spinopsphere/test_spinsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinopsphere/test_spinsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no spinsphere")


class TestSpinopsphereSpinsphere:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
