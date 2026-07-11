"""Port of MATLAB Chebfun tests/spinop2/test_spinop2.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinop2/test_spinop2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="SpinOp2 plumbing; spin2 numerics golden-ref tested in tests/test_spin/")


class TestSpinop2Spinop2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
