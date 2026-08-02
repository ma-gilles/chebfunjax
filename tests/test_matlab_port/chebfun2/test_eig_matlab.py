"""Port of MATLAB Chebfun tests/chebfun2/test_eig.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_eig.m
Chebfun commit: 7574c77

pass(1) checks the eigen-residual f*V = V*D; MATLAB uses
cheb.gallery2('challenge') (not ported), so an equivalent smooth
non-separable kernel exp(x*t) is used at the same 1e4*eps tolerance.
pass(2) checks the domain-square error.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.utils.quadrature import chebpts, chebweights

TOL = 1e4 * 2.220446049250313e-16


class TestChebfun2Eig:
    def test_pass1_eigen_residual(self):
        f = chebfun2(lambda x, t: jnp.exp(x * t))
        lam, V, xg = f.eig(return_functions=True)
        lam = np.asarray(lam)
        V = np.asarray(V)
        xg = np.asarray(xg)
        n = len(xg)
        w = np.asarray(chebweights(n), dtype=np.float64)
        pts = np.asarray(chebpts(n), dtype=np.float64)
        assert np.max(np.abs(pts - xg)) < 1e-14
        X, T = np.meshgrid(xg, xg, indexing="ij")
        F = np.asarray(f(jnp.asarray(X), jnp.asarray(T)))
        A = F * w[None, :]
        res = np.max(np.abs(A @ V - V * lam[None, :]))
        assert res < TOL

    def test_pass2_domain_error(self):
        f = chebfun2(lambda x, y: x + y, domain=(-1.0, 1.0, -2.0, 2.0))
        with pytest.raises(ValueError):
            f.eig()
