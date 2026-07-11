"""Port of MATLAB Chebfun tests/chebfun/test_constructor_singfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_singfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="'exps'/'blowup' ctor flags do not exist at the chebfun level (Singfun tested separately)")


class TestChebfunConstructorSingfun:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
