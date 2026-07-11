"""Port of MATLAB Chebfun tests/chebop/test_uminusOp.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_uminusOp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop unary minus not implemented")


class TestChebopUminusOp:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
