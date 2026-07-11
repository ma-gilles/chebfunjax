"""Port of MATLAB Chebfun tests/chebfun2/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="most ctor syntaxes tested (strings, coefficient matrices, values arrays, 'coeffs'/'trig' flags) do not exist on Chebfun2.from_function")


class TestChebfun2Constructor:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
