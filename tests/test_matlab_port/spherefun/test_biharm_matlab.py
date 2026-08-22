"""Port of MATLAB Chebfun tests/spherefun/test_biharm.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_biharm.m
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


class TestSpherefunBiharm:
    def test_all_matlab_assertions(self):
        tol = 1e4 * 100 * TOL2
        from chebfunjax.utils.quadrature import trigpts
        m = 6
        lam = np.pi * np.array(trigpts(m)[0])
        th = np.linspace(0.0, np.pi, m)
        L, T = np.meshgrid(lam, th)
        for ell in (1, 2, 4, 5):
            for mm in range(ell + 1):
                f = Spherefun.sphharm(ell, mm)
                lap2 = f.biharm()
                fac = (ell * (ell + 1)) ** 2
                F = fac * np.asarray(f(jnp.asarray(L), jnp.asarray(T)))
                A = np.asarray(lap2.fevalm(jnp.asarray(lam),
                                           jnp.asarray(th)))
                assert np.max(np.abs(F - A)) < fac * tol
