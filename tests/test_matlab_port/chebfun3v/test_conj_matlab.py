"""Port of MATLAB Chebfun tests/chebfun3v/test_conj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun3v: 'conj' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun3vConj:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
