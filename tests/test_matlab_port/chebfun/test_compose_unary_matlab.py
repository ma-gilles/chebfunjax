"""Port of MATLAB Chebfun tests/chebfun/test_compose_unary.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_compose_unary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="compose(f, op) unary composition is covered method-by-method (exp/log/sqrt/erf ports)")


class TestChebfunComposeUnary:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
