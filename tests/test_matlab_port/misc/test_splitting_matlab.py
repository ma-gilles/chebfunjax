"""Port of MATLAB Chebfun tests/misc/test_splitting.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_splitting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB global splitting() toggle has no counterpart; splitting=True kwarg is tested in tests/test_chebfun1d (SplittingOn)")


class TestMiscSplitting:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
