"""Port of MATLAB Chebfun tests/chebfun2/test_conj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no conj()")


class TestChebfun2Conj:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
