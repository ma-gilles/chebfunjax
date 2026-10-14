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

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform

jax.config.update("jax_enable_x64", True)

TOL = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
SUPPLEMENTAL_TOL = 1e3 * 2.220446049250313e-16


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
        state = init_matlab_twister(7)
        lam0, state = matlab_rand_uniform(state, ())
        th0, state = matlab_rand_uniform(state, ())
        u_at_point = u(lam0, th0)
        for v in (u.conj(), u.real()):
            v_at_point = v(lam0, th0)
            delta2 = sum(jnp.abs(a - b) ** 2
                         for a, b in zip(u_at_point, v_at_point))
            assert float(jnp.sqrt(delta2)) < TOL  # native pass(1)/(2)

        w = u.imag()
        assert w.norm() < TOL  # native pass(3)

        # Supplemental componentwise point checks retained from the earlier port.
        for v in (u.conj(), u.real()):
            for cu, cv in zip(u.components, v.components):
                assert abs(float(cu(lam0, th0)) - float(cv(lam0, th0))) < SUPPLEMENTAL_TOL
        assert _vnorm(w) < SUPPLEMENTAL_TOL
