"""Port of MATLAB Chebfun tests/chebop2/test_advectionDiffusion2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_advectionDiffusion2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Reference solution is a chebfun/pde15s (1-D method-of-lines) time trajectory, which chebfunjax does not implement.")


class TestChebop2Advectiondiffusion2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
