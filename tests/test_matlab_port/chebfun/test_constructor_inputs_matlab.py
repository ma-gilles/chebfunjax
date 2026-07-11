"""Port of MATLAB Chebfun tests/chebfun/test_constructor_inputs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_inputs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="numeric-matrix/string ctor inputs do not exist")


class TestChebfunConstructorInputs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
