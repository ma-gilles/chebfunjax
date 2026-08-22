"""Port of MATLAB Chebfun tests/spherefun/test_minandmax2est.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

TOL2 = 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


class TestSpherefunMinandmax2est:
    def test_all_matlab_assertions(self):
        tol = 1e3 * 100 * TOL2
        f = _sph(lambda x, y, z: z)
        mM = np.asarray(f.minandmax2est())
        assert np.linalg.norm(mM - np.array([-1.0, 1.0])) < tol
