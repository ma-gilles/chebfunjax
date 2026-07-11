"""Port of MATLAB Chebfun tests/chebop/test_nonlinSys2_C1.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSys2_C1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax chebop is scalar-only (no systems of ODEs / chebmatrix operators)")


class TestChebopNonlinsys2C1:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
