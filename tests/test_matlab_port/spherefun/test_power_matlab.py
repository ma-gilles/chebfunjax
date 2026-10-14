"""Port of MATLAB Chebfun tests/spherefun/test_power.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


class TestSpherefunPower:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = _sph(lambda x, y, z: z)
        g = _sph(lambda x, y, z: z ** 2)
        # MATLAB norm(A, inf) is the matrix infinity norm (maximum absolute
        # row sum), not the maximum absolute entry.
        assert np.linalg.norm(np.asarray(f.sample(100, 100)) ** 2
                              - np.asarray(g.sample(100, 100)), ord=np.inf) < tol
        # Supplementary control for scalar-left power, newly implemented in
        # the Python API and not present in the pinned MATLAB test.
        assert np.max(np.abs(np.asarray((2 ** f).sample(100, 100))
                             - 2 ** np.asarray(f.sample(100, 100)))) < tol
        f = _sph(lambda x, y, z: jnp.cos(x * y * z))
        g = _sph(lambda x, y, z: jnp.cos(x * y * z)
                 ** jnp.cos(x * y * z))
        # MATLAB norm(A) for a matrix is its spectral 2-norm.
        assert np.linalg.norm(np.asarray((f ** f).sample(100, 100)
                                         - g.sample(100, 100)), ord=2) < tol
