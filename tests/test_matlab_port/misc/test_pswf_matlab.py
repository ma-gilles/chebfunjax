"""Port of MATLAB Chebfun tests/misc/test_pswf.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_pswf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks chebfun-valued PSWFs incl. WolframAlpha point values; chebfunjax pswf returns grid values (NOT YET PORTED assertion-for-assertion)")


class TestMiscPswf:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
