"""Port of MATLAB Chebfun tests/chebfun3/test_sin.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_sin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no composition ops (sin(f))")


class TestChebfun3Sin:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
