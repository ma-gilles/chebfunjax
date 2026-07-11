"""Port of MATLAB Chebfun tests/chebop/test_eigs_schrodinger.m (Fable 5).

Harmonic oscillator -u'' + x^2 u: eigenvalues 2k+1.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_schrodinger.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop import Chebop


class TestChebopEigsSchrodinger:
    def test_harmonic_oscillator_spectrum(self):
        H = Chebop(lambda x, u: -u.diff(2) + x * x * u,
                   domain=(-8.0, 8.0))
        H.lbc = 0.0
        H.rbc = 0.0
        lam = H.eigs(k=6)
        lam = lam[0] if isinstance(lam, tuple) else lam
        lam = np.sort(np.real(np.asarray(lam)))
        exact = 2 * np.arange(6) + 1.0
        assert float(np.max(np.abs(lam - exact))) < 1e-6
