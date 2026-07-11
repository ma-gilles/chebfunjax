"""Port of MATLAB Chebfun tests/chebfun3/test_root.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_root.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no root/roots")


class TestChebfun3Root:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
