"""Port of MATLAB Chebfun tests/spherefun/test_cdr.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_cdr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no cdr accessor")


class TestSpherefunCdr:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
