"""Port of MATLAB Chebfun tests/chebfun/test_constructor_unbndfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_unbndfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="unbounded-domain ctor tested in unbndfun ports; chebfun-level assertions use MATLAB-only syntaxes")


class TestChebfunConstructorUnbndfun:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
