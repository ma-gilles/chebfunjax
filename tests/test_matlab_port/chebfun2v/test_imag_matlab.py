"""Port of MATLAB Chebfun tests/chebfun2v/test_imag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun2v: 'imag' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun2vImag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
