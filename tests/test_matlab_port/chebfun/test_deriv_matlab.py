"""Port of MATLAB Chebfun tests/chebfun/test_deriv.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_deriv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no deriv(f, x) convenience (diff + feval covers it)")


class TestChebfunDeriv:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
