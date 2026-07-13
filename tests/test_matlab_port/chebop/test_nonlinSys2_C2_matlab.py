"""Port of MATLAB Chebfun tests/chebop/test_nonlinSys2_C2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSys2_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB chebmatrix u{1}/u{2} cell-indexing NOTATION for the same system ported in test_nonlinSys1_C2_matlab.py (multi-argument form); the cell syntax itself is MATLAB-specific")


class TestChebopNonlinsys2C2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
