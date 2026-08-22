"""Port of MATLAB Chebfun tests/spherefunv/test_conj_imag_real.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_conj_imag_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)




class TestSpherefunvConjImagReal:
    def test_all_matlab_assertions(self):
        f = _sph(lambda x, y, z: jnp.cos((x + 0.1) * y * z))
        u = f.gradient()
        rng = np.random.RandomState(7)
        lam0, th0 = float(rng.rand()), float(rng.rand())
        L, T = jnp.asarray(lam0), jnp.asarray(th0)
        for v in (u.conj(), u.real()):
            for cu, cv in zip(u.components, v.components):
                assert abs(float(cu(L, T)) - float(cv(L, T))) < TOL
        w = u.imag()
        assert _vnorm(w) < TOL
