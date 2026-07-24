"""Port of MATLAB Chebfun tests/chebtech/test_chebTcoeffs2chebUcoeffs.m (Opus 4.8).

MATLAB ``chebtech.chebTcoeffs2chebUcoeffs`` converts first-kind Chebyshev (T)
coefficients to second-kind Chebyshev (U) coefficients.  chebfunjax now
provides this as a static method on both tech classes (via the identity
``T_n = (1/2)(U_n - U_{n-2})``), so every assertion is exercised.

Provenance
----------
MATLAB source : tests/chebtech/test_chebTcoeffs2chebUcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebTcoeffs2chebUcoeffs:
    def test_empty(self):
        # pass(1): isempty(chebTcoeffs2chebUcoeffs([])).
        out = Chebtech2.chebTcoeffs2chebUcoeffs(jnp.array([]))
        assert np.asarray(out).size == 0

    def test_column_vector(self):
        # pass(2): cT -> cU for 1 + x + x^2 + x^3 + x^4.
        cT = jnp.array([1.875, 1.75, 1.0, 0.25, 0.125])
        cU = Chebtech2.chebTcoeffs2chebUcoeffs(cT)
        cU_exact = jnp.array([1.375, 0.75, 0.4375, 0.125, 0.0625])
        assert _ninf(cU - cU_exact) < 10 * EPS

    def test_matrix(self):
        # pass(3): eye(5) -> recurrence matrix.
        cU = Chebtech2.chebTcoeffs2chebUcoeffs(jnp.eye(5))
        cU_exact = np.diag(0.5 * np.ones(5)) + np.diag(-0.5 * np.ones(3), 2)
        cU_exact[0, 0] = 1.0
        assert _ninf(np.asarray(cU) - cU_exact) < 10 * EPS
