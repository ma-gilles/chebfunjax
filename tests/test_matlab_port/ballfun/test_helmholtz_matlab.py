"""Port of MATLAB Chebfun tests/ballfun/test_helmholtz.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Ballfun has no Helmholtz solver (Poisson only)")


class TestBallfunHelmholtz:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
