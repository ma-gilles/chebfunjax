"""Port of MATLAB Chebfun tests/operatorBlock/test_isNotMultOrDiff.m (Fable 5).

Provenance
----------
MATLAB source : tests/operatorBlock/test_isNotMultOrDiff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax operator blocks are internal (chebfunjax.operators.blocks); the public chebop surface is tested in the chebop ports")


class TestOperatorblockIsnotmultordiff:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
