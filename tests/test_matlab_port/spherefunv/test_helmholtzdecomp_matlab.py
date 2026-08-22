"""Port of MATLAB Chebfun tests/spherefunv/test_helmholtzdecomp.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_helmholtzdecomp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

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


class TestSpherefunvHelmholtzdecomp:
    def test_all_matlab_assertions(self):
        tol = 1e5 * 2.220446049250313e-16
        cases = [
            (lambda x, y, z: jnp.cos(x * y * z),
             lambda x, y, z: jnp.sin(x + 0.1 * y + 5 * z ** 2),
             lambda x, y, z: x * y * z),
            (lambda x, y, z: jnp.sin((x - 0.1) * y * z),
             lambda x, y, z: jnp.cos(x + 0.5 * y - z ** 2),
             lambda x, y, z: -x * y ** 2 * z),
        ]
        for c1, c2, c3 in cases:
            f = Spherefunv(_sph(c1), _sph(c2), _sph(c3)).tangent()
            u, v = f.helmholtzdecomp()
            # f = grad(u) + curl(v) with curl(v) = n x grad(v).
            resid = Spherefunv(*[
                f.components[k] - u.gradient().components[k]
                - Spherefunv(
                    _sph(lambda x, y, z: x),
                    _sph(lambda x, y, z: y),
                    _sph(lambda x, y, z: z)).cross(
                        v.gradient()).components[k]
                for k in range(3)])
            assert _vnorm(resid) < 1e5 * tol
        # pass(4): empty in, empty out.
        u, v = Spherefunv.empty().helmholtzdecomp()
        assert u is None and v is None
