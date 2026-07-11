"""Port of MATLAB Chebfun tests/chebop/test_eigs_basic.m (Fable 5).

-u'' on [0, pi] with Dirichlet BCs: eigenvalues k^2, k = 1..10.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_basic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop

TOL_VALS = 1e-8


class TestChebopEigsBasic:
    def test_dirichlet_laplacian_eigenvalues(self):
        L = Chebop(lambda x, u: -u.diff(2), domain=(0.0, float(np.pi)))
        L.lbc = 0.0
        L.rbc = 0.0
        lam = L.eigs(k=10)
        lam = lam[0] if isinstance(lam, tuple) else lam
        lam = np.sort(np.real(np.asarray(lam)))
        exact = np.arange(1, 11, dtype=float) ** 2
        assert float(np.max(np.abs(lam - exact))) < TOL_VALS * exact[-1]

    def test_eigenfunction_residual(self):
        L = Chebop(lambda x, u: -u.diff(2), domain=(0.0, float(np.pi)))
        L.lbc = 0.0
        L.rbc = 0.0
        out = L.eigs(k=3)
        if not (isinstance(out, tuple) and len(out) >= 2):
            pytest.skip("eigs does not return eigenfunctions")
        lam, V = out[0], out[1]
        lam = np.asarray(lam)
        xs = jnp.asarray(np.linspace(0.2, np.pi - 0.2, 25))
        for i, v in enumerate(V if isinstance(V, list) else V):
            res = -v.diff(2)(xs) - float(np.real(lam[i])) * v(xs)
            scale = float(jnp.max(jnp.abs(v(xs))))
            assert float(jnp.max(jnp.abs(res))) < 1e-6 * max(
                scale * float(np.real(lam[i])), 1.0)
