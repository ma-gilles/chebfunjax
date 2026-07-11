"""Port of MATLAB Chebfun tests/spherefun/test_biharm.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_biharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no biharm")


class TestSpherefunBiharm:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
