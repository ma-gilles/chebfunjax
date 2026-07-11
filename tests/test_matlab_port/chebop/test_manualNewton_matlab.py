"""Port of MATLAB Chebfun tests/chebop/test_manualNewton.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_manualNewton.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="manual Newton stepping interface not exposed")


class TestChebopManualnewton:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
