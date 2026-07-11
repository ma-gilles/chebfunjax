"""Port of MATLAB Chebfun tests/misc/test_pswfpts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_pswfpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax pswfpts exists; MATLAB test needs pswf chebfun machinery (NOT YET PORTED assertion-for-assertion)")


class TestMiscPswfpts:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
