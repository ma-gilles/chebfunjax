"""Port of MATLAB Chebfun tests/chebfun3v/test_jacobian.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_jacobian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun3v: 'jacobian' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun3vJacobian:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
