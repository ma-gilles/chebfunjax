"""Port of MATLAB Chebfun tests/chebfun/test_realpow.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_realpow.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no realpow")


class TestChebfunRealpow:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
