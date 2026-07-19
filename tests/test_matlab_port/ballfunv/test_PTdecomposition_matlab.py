"""Port of MATLAB Chebfun tests/ballfunv/test_PTdecomposition.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_PTdecomposition.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Needs the poloidal-toroidal machinery (ballfunv.PT2ballfunv / PTdecomposition) which chebfunjax does not implement.")


class TestBallfunvPtdecomposition:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
