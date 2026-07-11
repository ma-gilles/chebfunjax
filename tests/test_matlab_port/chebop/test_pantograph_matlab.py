"""Port of MATLAB Chebfun tests/chebop/test_pantograph.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_pantograph.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="pantograph (delay) equations not supported")


class TestChebopPantograph:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
