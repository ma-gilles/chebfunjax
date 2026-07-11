"""Port of MATLAB Chebfun tests/misc/test_blowup.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_blowup.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB global blowup() toggle has no counterpart; blowup is a constructor concern (Singfun exponents) tested in singfun ports")


class TestMiscBlowup:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
