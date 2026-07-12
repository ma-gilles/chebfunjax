"""Port of MATLAB Chebfun tests/chebop/test_eigs_system.m (Fable 5).

FIXED: eigenvalue problems for systems of ODEs added in the Fable 5
audit (block collocation + generalized eig with BC rows zeroed in B,
two-resolution agreement filter for spurious discrete modes).  The
piecewise-domain case remains a documented skip.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_system.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop import Chebop


class TestChebopEigsSystem:
    def test_maxwell_inspired_system(self):
        d = (0.0, np.pi)
        A = Chebop(lambda x, u, v: [-u + v.diff(), u.diff()], d)
        A.lbc = lambda u, v: u
        A.rbc = lambda u, v: u
        _, lam = A.eigs(k=5)
        lam = np.sort(np.abs(np.asarray(lam)))
        correct = np.sort(np.abs(np.array([
            0,
            -0.5 + np.sqrt(3) / 2 * 1j,
            -0.5 - np.sqrt(3) / 2 * 1j,
            -0.5 + np.sqrt(15) / 2 * 1j,
            -0.5 - np.sqrt(15) / 2 * 1j])))
        assert np.max(np.abs(lam - correct)) < 1e-10
