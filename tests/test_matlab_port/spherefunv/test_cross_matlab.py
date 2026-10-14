"""Port of MATLAB Chebfun tests/spherefunv/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
SUPPLEMENTAL_EPS_TOL = 1e3 * 2.220446049250313e-16


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


class TestSpherefunvCross:
    def test_all_matlab_assertions(self):
        # pass(1): empty in, empty out.
        e = Spherefunv.empty()
        empty_cross = e.cross(e)
        assert empty_cross.isempty()  # native pass(1)
        # pass(2): u x u = 0.
        f = _sph(lambda x, y, z: jnp.cos((x + 0.1) * y * z))
        u = f.gradient()
        self_cross = u.cross(u)
        assert self_cross.norm() < TOL  # native pass(2)
        # pass(3): n x (grad f x grad g) is tangent-normal algebra;
        # the cross of two tangent fields is normal, so crossing with
        # the normal again vanishes... n x w where w || n gives 0.
        g = _sph(lambda x, y, z: jnp.sin(y * z))
        v = g.gradient()
        w = u.cross(v)
        nrml = Spherefunv(_sph(lambda x, y, z: x),
                          _sph(lambda x, y, z: y),
                          _sph(lambda x, y, z: z))
        normal_cross = nrml.cross(w)
        assert normal_cross.norm() < TOL  # native pass(3)

        # Supplemental sampled checks retained from the earlier port.
        assert _vnorm(self_cross) < 1e4 * SUPPLEMENTAL_EPS_TOL
        assert _vnorm(normal_cross) < 1e5 * SUPPLEMENTAL_EPS_TOL
