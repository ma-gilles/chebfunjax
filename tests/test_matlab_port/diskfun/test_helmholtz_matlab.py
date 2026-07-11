"""Port of MATLAB Chebfun tests/diskfun/test_helmholtz.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Diskfun has no Helmholtz solver (Poisson only)")


class TestDiskfunHelmholtz:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
