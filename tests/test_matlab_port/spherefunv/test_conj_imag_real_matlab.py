"""Port of MATLAB Chebfun tests/spherefunv/test_conj_imag_real.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_conj_imag_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefunv real/imag/conj exist (added Fable 5), but this MATLAB test builds u=grad(f), a 3-Cartesian-component field; chebfunjax Spherefunv is a 2-component (intrinsic tangent) representation, so the port needs a 3-component Spherefunv overhaul. Methods are exercised in tests/test_spherefunv core instead.")


class TestSpherefunvConjImagReal:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
