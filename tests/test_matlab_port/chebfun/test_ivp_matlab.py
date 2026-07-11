"""Port of MATLAB Chebfun tests/chebfun/test_ivp.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_ivp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB ode113/15s/45 chebfun wrappers; chebop IVP routing tested in operators ports")


class TestChebfunIvp:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
