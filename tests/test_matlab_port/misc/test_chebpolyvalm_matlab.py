"""Port of MATLAB Chebfun tests/misc/test_chebpolyvalm.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_chebpolyvalm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebpolyvalm (matrix polynomial evaluation)")


class TestMiscChebpolyvalm:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
