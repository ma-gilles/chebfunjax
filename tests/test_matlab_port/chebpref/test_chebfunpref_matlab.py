"""Port of MATLAB Chebfun tests/chebpref/test_chebfunpref.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebpref/test_chebfunpref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfunpref preference objects; eps/max_length kwargs are tested in the factory unit tests")


class TestChebprefChebfunpref:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
