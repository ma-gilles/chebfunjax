"""Port of MATLAB Chebfun tests/chebop/test_linearize.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_linearize.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="linearize() not exposed publicly")


class TestChebopLinearize:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
