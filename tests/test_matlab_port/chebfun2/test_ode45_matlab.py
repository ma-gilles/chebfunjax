"""Port of MATLAB Chebfun tests/chebfun2/test_ode45.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_ode45.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfun2-based ode45 phase-plane interface")


class TestChebfun2Ode45:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
