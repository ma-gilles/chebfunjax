"""Port of MATLAB Chebfun tests/chebop/test_minres.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_minres.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="operator minres not implemented")


class TestChebopMinres:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
