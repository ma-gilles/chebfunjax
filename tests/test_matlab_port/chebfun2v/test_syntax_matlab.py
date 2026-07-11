"""Port of MATLAB Chebfun tests/chebfun2v/test_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'syntax' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vSyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
