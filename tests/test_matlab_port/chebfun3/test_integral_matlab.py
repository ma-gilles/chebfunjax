"""Port of MATLAB Chebfun tests/chebfun3/test_integral.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no line-integral integral(f, curve)")


class TestChebfun3Integral:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
