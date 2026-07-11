"""Port of MATLAB Chebfun tests/misc/test_gpr.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_gpr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks chebfun-valued GPR outputs; chebfunjax gpr returns dict of arrays, covered by unit tests (NOT YET PORTED assertion-for-assertion)")


class TestMiscGpr:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
