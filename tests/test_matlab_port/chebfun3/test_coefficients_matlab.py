"""Port of MATLAB Chebfun tests/chebfun3/test_coefficients.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_coefficients.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no coefficient accessors")


class TestChebfun3Coefficients:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
