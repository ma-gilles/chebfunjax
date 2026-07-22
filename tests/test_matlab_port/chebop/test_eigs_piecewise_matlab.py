"""Port of MATLAB Chebfun tests/chebop/test_eigs_piecewise.m (Fable 5).

Generalized eigenvalue problem ``L u = lambda M u`` where the mass operator
``M`` has a discontinuous coefficient (an indicator bump).  The coefficient
jumps induce interior breakpoints; the pencil is assembled block-by-block
with continuity rows in both matrices.  From Chebfun issue #1074.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_piecewise.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402

# tol = 1e4 * bvpTol, bvpTol = 5e-13.
TOL = 1e4 * 5e-13


class TestChebopEigsPiecewise:
    def test_indicator_mass_pencil(self):
        ep = 0.25
        x = chebfun(lambda t: t, domain=(-1.0, 1.0))
        F = (x.abs() < ep) * (1.0 / (2 * ep))       # bump: 1/(2*ep) on |x|<ep
        L = Chebop(lambda xx, u: u.diff(2), (-1.0, 1.0))
        L.lbc = 0.0
        L.rbc = 0.0
        M = Chebop(lambda xx, u: F * u, (-1.0, 1.0))

        _, ec = L.eigs_generalized(M, k=6)
        ec = np.sort(np.real(np.asarray(ec)))

        ref = np.sort(1.0e2 * np.array([
            -4.987961490158024,
            -3.211336584876317,
            -1.829388534636815,
            -0.841876384196023,
            -0.247291299790162,
            -0.023950791540263,
        ]))
        assert ec.shape == ref.shape
        assert np.max(np.abs(ec - ref)) < TOL
