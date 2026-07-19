"""Port of MATLAB Chebfun tests/spinop/test_spin.m (Fable 5).

Spinop + spin (ETDRK4, Kassam-Trefethen phi functions) added in the
Fable 5 audit (Big-Three spinop directive).  The AC (Allen-Cahn)
self-convergence case passes at 1.4e-9.  The CH (Cahn-Hilliard) case
used to blow up at t ~ 30; the root cause was a roundoff-seeded
conjugate-antisymmetric (imaginary-u) mode growing unsaturated under
CH's linear growth band, since the nonlinear term only sees
np.real(ifft(.)).  Fixed by re-Hermitianizing the Fourier state each
step; CH now self-converges at ~9e-8 with the physical plateau
max|u| = 0.9998.

Provenance
----------
MATLAB source : tests/spinop/test_spin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.spinop import Spinop, spin

TOL = 1e-6


class TestSpin:
    def test_allen_cahn_self_convergence(self):
        S = Spinop("AC")
        u = spin(S, 256, 1e-1)
        v = spin(S, 256, 5e-2)
        xs = jnp.asarray(np.linspace(0.01, 6.2, 60))
        rel = float(jnp.max(jnp.abs(u(xs) - v(xs)))) \
            / float(jnp.max(jnp.abs(v(xs))))
        assert rel < TOL

    def test_cahn_hilliard_self_convergence(self):
        S = Spinop("CH")
        u = spin(S, 256, 2e-2)
        v = spin(S, 256, 1e-2)
        xs = jnp.asarray(np.linspace(-0.99, 0.99, 60))
        rel = float(jnp.max(jnp.abs(u(xs) - v(xs)))) \
            / float(jnp.max(jnp.abs(v(xs))))
        assert np.isfinite(rel) and rel < TOL
        # physical CH plateau is +/-1, not a blowup
        assert float(jnp.max(jnp.abs(v(xs)))) < 1.5
