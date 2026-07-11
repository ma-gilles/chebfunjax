"""Port of MATLAB Chebfun tests/spherefun/test_HelmholtzSolver.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_HelmholtzSolver.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no Helmholtz solver (Poisson only)")


class TestSpherefunHelmholtzsolver:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
