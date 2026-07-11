"""Port of MATLAB Chebfun tests/misc/test_pde15s.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_pde15s.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test uses chebfun/chebmatrix PDE syntax; chebfunjax pde15s covered by tests/test_coverage (NOT YET PORTED assertion-for-assertion)")


class TestMiscPde15s:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
