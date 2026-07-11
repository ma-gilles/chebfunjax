"""Port of MATLAB Chebfun tests/chebop/test_multipleOutputs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_multipleOutputs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="multiple-output solve diagnostics not exposed")


class TestChebopMultipleoutputs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
