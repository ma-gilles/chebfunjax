"""Port of MATLAB Chebfun tests/spinscheme/test_startMultistep.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinscheme/test_startMultistep.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax exposes only ETDRK4 (scheme selection internal); ETDRK4 parity is golden-ref tested in tests/test_spin")


class TestSpinschemeStartmultistep:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
