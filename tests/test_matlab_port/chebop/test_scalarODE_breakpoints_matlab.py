"""Port of MATLAB Chebfun tests/chebop/test_scalarODE_breakpoints.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE_breakpoints.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="piecewise-domain chebop solve not implemented")


class TestChebopScalarodeBreakpoints:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
