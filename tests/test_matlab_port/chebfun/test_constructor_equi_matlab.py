"""Port of MATLAB Chebfun tests/chebfun/test_constructor_equi.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_equi.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="'equi' flag does not exist")


class TestChebfunConstructorEqui:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
