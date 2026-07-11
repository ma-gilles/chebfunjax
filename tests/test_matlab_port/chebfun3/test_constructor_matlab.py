"""Port of MATLAB Chebfun tests/chebfun3/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="most ctor syntaxes (strings, arrays, flags) do not exist on Chebfun3.from_function")


class TestChebfun3Constructor:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
