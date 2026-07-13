"""Port of MATLAB Chebfun tests/chebfun/test_aaatrig.m (Fable 5).

FIXED: aaatrig existed but every complex-valued approximation was
silently broken -- four SVD null-vector sites used rows of numpy's
Vh without conjugation.  With the conjugate fix the MATLAB cases
pass at machine precision for both barycentric forms.

Provenance
----------
MATLAB source : tests/chebfun/test_aaatrig.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.utils.aaa import aaatrig

TOL = 1e-10
Z = np.linspace(0, 2 * np.pi, 1000)


@pytest.mark.parametrize("form", ["odd", "even"])
class TestChebfunAaatrig:
    def test_smooth_and_controls(self, form):
        F = np.exp(np.cos(Z))
        out = aaatrig(F, Z, form=form)
        r, zj = out[0], np.asarray(out[4])
        # pass(1)
        assert float(np.max(np.abs(F - r(Z)))) < 2 * TOL
        m1 = len(zj)
        # pass(4): mmax control
        out2 = aaatrig(F, Z, form=form, mmax=m1 - 1)
        assert len(np.asarray(out2[4])) == m1 - 1
        # pass(5): tol control
        out3 = aaatrig(F, Z, form=form, tol=1e-3)
        assert len(np.asarray(out3[4])) < m1

    def test_rational_poles_zeros(self, form):
        # pass(6)-(8)
        F = np.sin(3 * Z) * np.cos(7 * Z) / np.sin(2 * Z - 1j)
        out = aaatrig(F, Z, form=form, cleanup=False)
        r, pol, zer = out[0], np.asarray(out[1]), np.asarray(out[3])
        assert float(np.max(np.abs(F - r(Z)))) < 10 * TOL
        assert float(np.min(np.abs(zer - np.pi / 2))) < TOL
        assert float(np.min(np.abs(
            np.real(pol - 0.5j) - np.pi / 2))) < TOL
