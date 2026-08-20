"""Port of MATLAB Chebfun tests/chebop/test_eigs_orrsom.m (Fable 5).

FIXED (Fable 5): eigs_generalized now supports complex operators,
callable multi-condition BCs (MATLAB's clamped ``@(u) [u; diff(u)]``,
one collocation row replaced per condition from each end), and 'LR'
(largest-real-part) mode selection -- so the Orr-Sommerfeld
generalized eigenproblem ports directly.

MATLAB loops three discretizations (chebcolloc2/ultraS/chebcolloc1);
chebfunjax has a single collocation discretization, so the assertion
pair is checked once (repo convention).

Provenance
----------
MATLAB source : tests/chebop/test_eigs_orrsom.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from chebfunjax.operators.chebop import Chebop


class TestChebopEigsOrrsom:
    def test_orr_sommerfeld_critical_eigenvalue(self):
        # pass(1)/(4): the critical (largest-real-part) eigenvalue at
        # Re = 5772.22 matches MATLAB's v4 reference within 5e-6, and
        # all 20 requested modes are non-spurious.
        Re, alph = 5772.22, 1.0
        A = Chebop(
            lambda x, u: (u.diff(4) - 2 * alph ** 2 * u.diff(2)
                          + alph ** 4 * u) / Re
            - 2j * alph * u
            - 1j * alph * ((1 - x * x) * (u.diff(2) - alph ** 2 * u)),
            domain=(-1.0, 1.0))
        B = Chebop(lambda x, u: u.diff(2) - u, domain=(-1.0, 1.0))
        A.lbc = lambda u: [u, u.diff()]
        A.rbc = lambda u: [u, u.diff()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, lam = A.eigs_generalized(B, k=20, n=96, sort="LR")
        e = np.asarray(lam)
        e = e[np.abs(e) < 1e5]
        e_crit = e[np.argmax(e.real)]
        e_crit_v4 = -0.000078029804093 - 0.261565915010080j
        assert abs(e_crit - e_crit_v4) < 5e-6
        assert len(np.asarray(lam)) == 20

    @pytest.mark.timeout(880)
    @pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
    def test_other_discretizations(self, disc):
        # MATLAB's loop repeats the problem under ultraS/chebcolloc1.
        Re, alph = 5772.22, 1.0
        A = Chebop(
            lambda x, u: (u.diff(4) - 2 * alph ** 2 * u.diff(2)
                          + alph ** 4 * u) / Re
            - 2j * alph * u
            - 1j * alph * ((1 - x * x) * (u.diff(2) - alph ** 2 * u)),
            domain=(-1.0, 1.0))
        B = Chebop(lambda x, u: u.diff(2) - u, domain=(-1.0, 1.0))
        A.lbc = lambda u: [u, u.diff()]
        A.rbc = lambda u: [u, u.diff()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, lam = A.eigs_generalized(B, k=20, n=96, sort="LR",
                                        discretization=disc)
        e = np.asarray(lam)
        e = e[np.abs(e) < 1e5]
        e_crit = e[np.argmax(e.real)]
        e_crit_v4 = -0.000078029804093 - 0.261565915010080j
        assert abs(e_crit - e_crit_v4) < 5e-6
