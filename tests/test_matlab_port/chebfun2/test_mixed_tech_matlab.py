"""Port of MATLAB Chebfun tests/chebfun2/test_mixed_tech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_mixed_tech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no trig/periodic tech option to mix")


class TestChebfun2Mixedtech:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
