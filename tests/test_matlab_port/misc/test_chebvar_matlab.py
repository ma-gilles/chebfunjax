"""Port of MATLAB Chebfun tests/misc/test_chebvar.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_chebvar.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB 'chebvar x' workspace magic has no Python counterpart; chebfun(lambda x: x) covers the semantics")


class TestMiscChebvar:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
