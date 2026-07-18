"""Port of MATLAB Chebfun tests/chebop/test_eigs_orrsom.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_eigs_orrsom.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Orr-Sommerfeld needs clamped BCs u = u' = 0 at both ends (a callable BC returning TWO conditions [u; diff(u)]); chebfunjax's generalized eigs (eigs_generalized) accepts only string ('dirichlet'/'neumann') or scalar BCs and raises 'unsupported bc' on a callable multi-condition BC, so this 4th-order generalized eigenproblem cannot be posed -- src gap")


class TestChebopEigsOrrsom:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
