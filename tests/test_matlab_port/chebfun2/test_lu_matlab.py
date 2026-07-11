"""Port of MATLAB Chebfun tests/chebfun2/test_lu.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_lu.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no lu() factorization")


class TestChebfun2Lu:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
