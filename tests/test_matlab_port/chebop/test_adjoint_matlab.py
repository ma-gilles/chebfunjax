"""Port of MATLAB Chebfun tests/chebop/test_adjoint.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_adjoint.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop has no adjoint()")


class TestChebopAdjoint:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
