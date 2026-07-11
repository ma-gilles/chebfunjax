"""Port of MATLAB Chebfun tests/chebfun3/test_gradient.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_gradient.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3.grad exists but the MATLAB test checks chebfun3v output and norms of components (chebfun3v ops absent)")


class TestChebfun3Gradient:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
