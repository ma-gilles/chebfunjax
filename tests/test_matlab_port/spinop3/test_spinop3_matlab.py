"""Port of MATLAB Chebfun tests/spinop3/test_spinop3.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinop3/test_spinop3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="SpinOp3 plumbing; spin3 numerics golden-ref tested in tests/test_spin/")


class TestSpinop3Spinop3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
