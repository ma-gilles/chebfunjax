"""Port of MATLAB Chebfun tests/ballfun/test_exp.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_exp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no exp composition")


class TestBallfunExp:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
