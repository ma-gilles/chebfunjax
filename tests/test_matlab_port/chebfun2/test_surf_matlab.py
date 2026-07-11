"""Port of MATLAB Chebfun tests/chebfun2/test_surf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_surf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot-output test; no MATLAB-compatible return values")


class TestChebfun2Surf:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
