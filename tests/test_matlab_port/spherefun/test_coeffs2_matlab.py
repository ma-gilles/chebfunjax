"""Port of MATLAB Chebfun tests/spherefun/test_coeffs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_coeffs2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

from ._cart import sph_lt

jax.config.update("jax_enable_x64", True)


def _err(X, c):
    X = np.asarray(X)
    c = np.asarray(c)
    if X.shape != c.shape:
        # embed both in a common centred frame
        m = max(X.shape[0], c.shape[0])
        n = max(X.shape[1], c.shape[1])

        def _pad(A):
            out = np.zeros((m, n), dtype=complex)
            r0 = (m - A.shape[0]) // 2
            c0 = (n - A.shape[1]) // 2
            out[r0:r0 + A.shape[0], c0:c0 + A.shape[1]] = A
            return out
        X, c = _pad(X), _pad(c)
    return np.linalg.norm(X - c)


class TestSpherefunCoeffs2:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = sph_lt(lambda lam, th: jnp.cos(th))
        assert _err(f.coeffs2(), np.array([[.5], [0], [.5]])) < tol           # pass(1)
        f = sph_lt(lambda lam, th: jnp.sin(th) * jnp.sin(lam))
        c = .25 * np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
        assert _err(f.coeffs2(), c) < tol                                    # pass(2)
        f = sph_lt(lambda lam, th: jnp.sin(th) * jnp.sin(lam) + jnp.cos(th))
        c = .25 * np.array([[-1, 2, 1], [0, 0, 0], [1, 2, -1]])
        assert _err(f.coeffs2(), c) < tol                                    # pass(3)
        f = sph_lt(lambda lam, th: jnp.sin(th) ** 2 * jnp.sin(lam) * jnp.cos(lam) + jnp.cos(th) ** 2)
        c = (1 / 16) * np.array([[-1j, 0, 4, 0, 1j], [0, 0, 0, 0, 0], [2j, 0, 8, 0, -2j],
                                 [0, 0, 0, 0, 0], [-1j, 0, 4, 0, 1j]])
        assert _err(f.coeffs2(), c) < tol                                    # pass(4)
        f = Spherefun.from_values(np.zeros((5, 4)))
        # MATLAB: coeffs2(f) - zeros(5, 4) (scalar-expanded) has norm 0
        assert np.linalg.norm(np.asarray(f.coeffs2())) == 0                  # pass(5)
