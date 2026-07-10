"""Port of MATLAB Chebfun tests/chebtech/test_mldivide.m (Fable 5).

Every assertion in the MATLAB file exercises the tech-level 'mldivide'
method on (mostly array-valued) chebtechs.  chebfunjax techs are
single-column and have no 'mldivide' method; quasimatrix least-squares/QR
live at the Chebfun level (chebfunjax.chebfun1d.linalg) and are tested
by tests/test_chebfun1d/test_linalg_matlab.py.

Provenance
----------
MATLAB source : tests/chebtech/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax techs have no mldivide method (single-column techs; "
    "quasimatrix linear algebra lives at the Chebfun level)"
)


class TestChebtechMldivide:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
