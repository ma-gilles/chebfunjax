"""Port of MATLAB Chebfun tests/chebop/test_ivp_chebmatrix_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_ivp_chebmatrix_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB chebmatrix cell-syntax variant of system IVPs; the functionality is ported in test_LorenzIVP_matlab.py")


class TestChebopIvpChebmatrixSyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
