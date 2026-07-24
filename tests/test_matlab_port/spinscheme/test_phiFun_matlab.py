"""Port of MATLAB Chebfun tests/spinscheme/test_phiFun.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinscheme/test_phiFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no public expinteg.phiFun; the exponential phi-function weights are computed inline in the ETDRK4 coefficient builders (spin.solver2d._compute_etdrk4_coeffs_2d, spin.solver3._compute_etdrk4_coeffs_3d, operators.spinop.spin) and are golden-ref validated end-to-end by the spin* self-convergence ports")


class TestSpinschemePhifun:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
