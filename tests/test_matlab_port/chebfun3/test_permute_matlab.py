"""Port of MATLAB Chebfun tests/chebfun3/test_permute.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_permute.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no permute")


class TestChebfun3Permute:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
