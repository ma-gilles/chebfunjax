"""Port of MATLAB Chebfun tests/ballfunv/test_HelmholtzDecomposition.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_HelmholtzDecomposition.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB HelmholtzDecomposition uses the poloidal-toroidal API (ballfunv.PT2ballfunv / PTdecomposition), which chebfunjax lacks; the existing Ballfunv.helmholtz_decomposition is a different (Hodge curl-free/div-free) split, not comparable to this test. Needs the PT subsystem.")


class TestBallfunvHelmholtzdecomposition:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
