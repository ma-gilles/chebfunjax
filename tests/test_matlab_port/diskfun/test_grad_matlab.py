"""Port of MATLAB Chebfun tests/diskfun/test_grad.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_grad.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Diskfun has no grad (diffx/diffy tested in diff port)")


class TestDiskfunGrad:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
