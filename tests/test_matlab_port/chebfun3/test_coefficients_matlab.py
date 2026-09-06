"""Port of MATLAB Chebfun tests/chebfun3/test_coefficients.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_coefficients.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebpoly
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.quadrature import chebpts

jax.config.update("jax_enable_x64", True)


class TestChebfun3Coefficients:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
        n = 10
        f, g, h = chebpoly(n), chebpoly(n), chebpoly(n)
        k = chebfun3(lambda x, y, z: f(x) * g(y) * h(z))
        X = np.array(k.chebcoeffs3())
        assert abs(X[n, n, n] - 1) < tol                                     # pass(1)
        X[n, n, n] -= 1
        assert np.linalg.norm(X.ravel()) < tol                               # pass(2)

        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        m, n2, p = f.length()
        x = np.asarray(chebpts(int(m), kind=2))
        y = np.asarray(chebpts(int(n2), kind=2))
        z = np.asarray(chebpts(int(p), kind=2))
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        vals = np.asarray(f(jnp.asarray(xx.ravel()), jnp.asarray(yy.ravel()),
                            jnp.asarray(zz.ravel()))).reshape(xx.shape)
        X = np.asarray(Chebfun3.coeffs2vals(f.chebcoeffs3()))
        assert np.linalg.norm(X.ravel() - vals.ravel()) < tol                # pass(3)
