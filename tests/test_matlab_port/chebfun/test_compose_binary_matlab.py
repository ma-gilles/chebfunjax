"""Port of MATLAB Chebfun tests/chebfun/test_compose_binary.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_compose_binary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no compose(f, op, g) binary composition")


class TestChebfunComposeBinary:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
