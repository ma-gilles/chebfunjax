"""Port of MATLAB Chebfun tests/chebfun/test_constructor_basic_periodic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_basic_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="string + 'periodic' ctor syntaxes do not exist")


class TestChebfunConstructorBasicPeriodic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
