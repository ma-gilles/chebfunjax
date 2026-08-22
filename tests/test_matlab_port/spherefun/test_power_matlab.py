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


class TestSpherefunPower:
    def test_all_matlab_assertions(self):
        tol = 1000 * 100 * TOL2
        f = _sph(lambda x, y, z: z)
        g = _sph(lambda x, y, z: z ** 2)
        assert np.max(np.abs(np.asarray(f.sample(100, 100)) ** 2
                             - np.asarray(g.sample(100, 100)))) < tol
        f = _sph(lambda x, y, z: jnp.cos(x * y * z))
        g = _sph(lambda x, y, z: jnp.cos(x * y * z)
                 ** jnp.cos(x * y * z))
        assert np.linalg.norm(np.asarray((f ** f).sample(100, 100))
                              - np.asarray(g.sample(100, 100))) < tol
