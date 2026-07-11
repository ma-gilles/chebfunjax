"""Port of MATLAB Chebfun tests/chebfun2/test_roots_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_roots_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="roots(f,g) two-function syntax lives on Chebfun2v; only marching-squares roots(f) exists")


class TestChebfun2Rootssyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
