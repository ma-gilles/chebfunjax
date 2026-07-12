"""Port of MATLAB Chebfun tests/chebop/test_eigs_system2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_eigs_system2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB chebmatrix u{1}/u{2} cell-indexing NOTATION for the same Maxwell-inspired eigenproblem already ported in test_eigs_system_matlab.py (multi-argument form); the cell syntax itself is MATLAB-specific")


class TestChebopEigsSystem2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
