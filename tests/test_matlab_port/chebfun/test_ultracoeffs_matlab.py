"""Port of MATLAB Chebfun tests/chebfun/test_ultracoeffs.m (Fable 5).

MATLAB's ``ultrapoly(0:n, lam) * c`` (quasimatrix times vector) is the
chebfun with Chebyshev coefficients ``sum_k c_k * ultrapoly(k, lam)``.

Provenance
----------
MATLAB source : tests/chebfun/test_ultracoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.polynomials import legpoly, ultrapoly

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * np.finfo(float).eps


def _combine(coeff_vectors, c):
    n = max(len(v) for v in coeff_vectors)
    total = np.zeros(n)
    for ck, v in zip(c, coeff_vectors):
        v = np.asarray(v)
        total[:len(v)] += ck * v
    return chebfun(jnp.asarray(total), coeffs=True)


class TestChebfunUltracoeffs:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(42)
        n, lam = 10, 0.1
        C = [ultrapoly(k, lam) for k in range(n + 1)]
        c = rng.rand(n + 1)
        err = np.max(np.abs(c - np.asarray(_combine(C, c).ultracoeffs(n + 1, lam))))
        assert err < TOL                                            # pass(1)

        n = 10
        L = [legpoly(k) for k in range(n + 1)]
        c = rng.rand(n + 1)
        err = np.max(np.abs(c - np.asarray(_combine(L, c).ultracoeffs(n + 1, 0.5))))
        assert err < TOL                                            # pass(2)

        f = chebfun(jnp.exp)
        err = np.max(np.abs(np.asarray(f.legcoeffs()) - np.asarray(f.ultracoeffs(0.5))))
        assert err < TOL                                            # pass(3)

        f = chebfun(jnp.exp)
        err = np.max(np.abs(np.asarray(f.chebcoeffs(kind=2))
                            - np.asarray(f.ultracoeffs(1.0))))
        assert err < TOL                                            # pass(4)
