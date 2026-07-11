"""Port of MATLAB Chebfun tests/chebfun2/test_complex.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="requires real/imag/conj on Chebfun2 (absent)")


class TestChebfun2Complex:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
