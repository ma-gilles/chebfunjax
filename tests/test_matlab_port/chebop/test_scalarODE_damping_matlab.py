"""Port of MATLAB Chebfun tests/chebop/test_scalarODE_damping.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE_damping.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB inspects info.normDelta damping diagnostics; solve() does not expose Newton step info")


class TestChebopScalarODE_damping:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
