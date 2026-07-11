"""Port of MATLAB Chebfun tests/chebop/test_multOutputs_simplify.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_multOutputs_simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="multiple-output simplify not exposed")


class TestChebopMultoutputsSimplify:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
