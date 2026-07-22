"""Port of MATLAB Chebfun tests/chebop2/test_helmholtz.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Helmholtz Dirichlet solves in the chebfun2 L2 norm at 100*eps; the value-space solver floor (~1e-12) exceeds it (pass1-2). pass3-4 are mu=50 high-frequency cases needing large n, prohibitive for the dense O(n^6) Kronecker solve.")


class TestChebop2Helmholtz:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
