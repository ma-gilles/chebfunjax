"""Port of MATLAB Chebfun tests/chebmatrix/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebmatrix (block operator/chebfun container) class")


class TestChebmatrixConstructor:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
