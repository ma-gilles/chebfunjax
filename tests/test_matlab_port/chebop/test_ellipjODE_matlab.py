"""Port of MATLAB Chebfun tests/chebop/test_ellipjODE.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_ellipjODE.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="nonlinear pendulum BVP needs ellipj-based exact solution machinery and N.init tuning beyond current Newton robustness")


class TestChebopEllipjODE:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
