"""Port of MATLAB Chebfun tests/spinscheme/test_startMultistep.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinscheme/test_startMultistep.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="test compares ETDRK4 vs the multistep PECEC736 (1D/2D) and LIRK4 vs IMEXBDF4 (sphere) via a 'scheme' kwarg with startMultistep bootstrapping; chebfunjax 1D/2D/3D spin exposes only ETDRK4 (no pecec736/lawson4) and offers no 'scheme' selector, so the multistep startMultistep path does not exist to exercise")


class TestSpinschemeStartmultistep:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
