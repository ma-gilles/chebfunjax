"""Port of MATLAB Chebfun tests/chebfun2/test_diag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_diag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no diag (diagonal chebfun extraction)")


class TestChebfun2Diag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
