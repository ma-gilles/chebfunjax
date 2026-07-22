"""Port of MATLAB Chebfun tests/chebop2/test_adaptivity.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_adaptivity.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires multi-condition BC syntax rbc=@(t,u)[u;diff(u)], a 3rd-order-in-x term diffx(u,3), and the length-controlled mldivide(N,0,nx,ny) adaptivity API; none exist in the value-space scalar Chebop2 solver.")


class TestChebop2Adaptivity:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
