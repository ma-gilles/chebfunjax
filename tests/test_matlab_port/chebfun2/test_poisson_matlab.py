"""Port of MATLAB Chebfun tests/chebfun2/test_poisson.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfun2.poisson fast solver (chebop2 covers Poisson separately)")


class TestChebfun2Poisson:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
