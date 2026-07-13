"""Port of MATLAB Chebfun tests/chebop/test_eigs_drum.m (Fable 5).

FIXED: generalized eigenvalue problems eigs(A, B) added in the
Fable 5 audit (basis-probed matrices, BC rows in A zeroed in B,
two-resolution spurious filter).  The drum frequencies are the
zeros of the Bessel function J0.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_drum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros

from chebfunjax.operators.chebop import Chebop


class TestChebopEigsDrum:
    def test_drum_frequencies_are_bessel_zeros(self):
        A = Chebop(lambda r, u: r * u.diff(2) + u.diff(),
                   (0.0, 1.0))
        A.lbc = "neumann"
        A.rbc = "dirichlet"
        B = Chebop(lambda r, u: r * u, (0.0, 1.0))
        _, lam = A.eigs_generalized(B, k=6)
        omega = np.sort(np.sqrt(-np.real(np.asarray(lam))))
        assert np.max(np.abs(omega - jn_zeros(0, 6))) < 1e-8
