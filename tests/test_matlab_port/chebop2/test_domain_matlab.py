"""Port of MATLAB Chebfun tests/chebop2/test_domain.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_domain.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Needs boundary-slice accessors u(:,y)/u(x,:) returning a 1-D chebfun for the BC-imposition checks, plus a Robin BC dbc=@(x,u) u+2.1*diff(u)-... (pass10).")


class TestChebop2Domain:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
