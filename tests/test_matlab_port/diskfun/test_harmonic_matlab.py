"""Port of MATLAB Chebfun tests/diskfun/test_harmonic.m (Fable 5).

FIXED: diskfun.harmonic added in the Fable 5 audit (Bessel-mode
eigenfunctions, Dirichlet and Neumann).

Provenance
----------
MATLAB source : tests/diskfun/test_harmonic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import jv

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 3e2 * np.finfo(float).eps
THS = jnp.asarray(np.linspace(-np.pi, np.pi, 13))
RS = jnp.asarray(np.linspace(0.0, 1.0, 9))
TT, RR = jnp.meshgrid(THS, RS, indexing="ij")


class TestDiskfunHarmonic:
    def test_closed_forms(self):
        # pass(1)-(3): known Bessel roots
        j53 = 15.7001740797116
        c = np.sqrt(2) / (np.sqrt(np.pi) * abs(jv(6, j53)))

        G = Diskfun.harmonic(5, 3)
        F = c * jv(5, j53 * np.asarray(RR)) * np.cos(5 * np.asarray(TT))
        assert float(jnp.max(jnp.abs(G(TT, RR) - jnp.asarray(F)))) \
            < 1e4 * TOL

        Gs = Diskfun.harmonic(-5, 3)
        Fs = c * jv(5, j53 * np.asarray(RR)) * np.sin(5 * np.asarray(TT))
        assert float(jnp.max(jnp.abs(Gs(TT, RR) - jnp.asarray(Fs)))) \
            < 1e4 * TOL

        j05 = 14.9309177084877
        c0 = np.sqrt(2) / (np.sqrt(2 * np.pi) * abs(jv(1, j05)))
        G0 = Diskfun.harmonic(0, 5)
        F0 = c0 * jv(0, j05 * np.asarray(RR)) * np.ones_like(
            np.asarray(TT))
        assert float(jnp.max(jnp.abs(G0(TT, RR) - jnp.asarray(F0)))) \
            < 1e4 * TOL

    def test_orthonormality(self):
        # pass(4)-(8)
        A = Diskfun.harmonic(29, 10)
        B = Diskfun.harmonic(6, 23)
        C = Diskfun.harmonic(-4, 3)
        assert abs(float((A * A).sum2()) - 1) < 1e5 * TOL
        assert abs(float((B * B).sum2()) - 1) < 1e5 * TOL
        assert abs(float((A * B).sum2())) < 1e5 * TOL
        assert abs(float((A * C).sum2())) < 1e5 * TOL
        assert abs(float((C * C).sum2()) - 1) < 1e5 * TOL

    def test_neumann_orthonormality(self):
        # pass(8)-(10)
        A = Diskfun.harmonic(10, 8, "neumann")
        B = Diskfun.harmonic(-4, 3, "neumann")
        assert abs(float((A * A).sum2()) - 1) < 1e5 * TOL
        assert abs(float((B * B).sum2()) - 1) < 1e5 * TOL
        assert abs(float((A * B).sum2())) < 1e5 * TOL
