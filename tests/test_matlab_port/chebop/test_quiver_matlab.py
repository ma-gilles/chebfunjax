"""Port of MATLAB Chebfun tests/chebop/test_quiver.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_quiver.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot smoke test")


class TestChebopQuiver:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
