"""Port of MATLAB Chebfun tests/chebop/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop N(x, u) direct evaluation not implemented")


class TestChebopFeval:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
