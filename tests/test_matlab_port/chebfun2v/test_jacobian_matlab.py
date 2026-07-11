"""Port of MATLAB Chebfun tests/chebfun2v/test_jacobian.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_jacobian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'jacobian' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vJacobian:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
