"""Port of MATLAB Chebfun tests/chebfun3t/test_ndf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3t/test_ndf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfun3t (full-tensor variant); Chebfun3 Tucker class is ported in chebfun3")


class TestChebfun3tNdf:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
