"""Port of MATLAB Chebfun tests/misc/test_conformal2.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_conformal2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks chebfun-valued rectangle maps; chebfunjax conformal2 covered by tests/test_utils/test_final5.py")


class TestMiscConformal2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
