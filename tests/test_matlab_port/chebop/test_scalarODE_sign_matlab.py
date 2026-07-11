"""Port of MATLAB Chebfun tests/chebop/test_scalarODE_sign.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="sign-coefficient ODEs need piecewise ops")


class TestChebopScalarodeSign:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
