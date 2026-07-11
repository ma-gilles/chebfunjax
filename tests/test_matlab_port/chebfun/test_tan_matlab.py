"""Port of MATLAB Chebfun tests/chebfun/test_tan.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_tan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test builds tan with blowup exponents at poles ('exps'); no chebfun-level blowup")


class TestChebfunTan:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
