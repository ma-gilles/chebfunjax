"""Port of MATLAB Chebfun tests/chebop/test_linearInit.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_linearInit.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="linear-solve init path internal; covered by scalarODE ports")


class TestChebopLinearinit:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
