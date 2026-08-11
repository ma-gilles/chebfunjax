"""Port of MATLAB Chebfun tests/linop/test_svds.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_svds.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="MATLAB's linop/svds builds the singular triplets from the "
           "generalized eigenproblem for [0 L*; L 0], so it needs "
           "linopAdjoint on a linop (the formal adjoint plus the adjoint "
           "boundary conditions derived from the boundary bilinear form). "
           "chebfunjax has a scalar-chebop adjoint in "
           "src/chebfunjax/operators/adjoint.py, but BlockLinop has no "
           "linopAdjoint and no svds, so neither the singular values "
           "(pass 1, 3) nor the L*V = U*S check (pass 2, 4) can be "
           "computed. Porting linopAdjoint for BlockLinop unblocks this "
           "file together with test_linopAdjoint_matlab.py.")


class TestLinopSvds:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
