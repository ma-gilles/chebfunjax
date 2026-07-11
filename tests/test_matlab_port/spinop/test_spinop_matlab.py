"""Port of MATLAB Chebfun tests/spinop/test_spinop.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinop/test_spinop.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="SpinOp preset/timestep plumbing; ETDRK4 numerics are golden-ref tested in tests/test_spin/")


class TestSpinopSpinop:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
