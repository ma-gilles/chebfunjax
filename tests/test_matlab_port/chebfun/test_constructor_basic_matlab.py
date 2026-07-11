"""Port of MATLAB Chebfun tests/chebfun/test_constructor_basic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_basic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="string ctor syntaxes ('x', 'sin(x)') do not exist")


class TestChebfunConstructorBasic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
