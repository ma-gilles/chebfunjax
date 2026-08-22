"""Port of MATLAB Chebfun tests/spherefunv/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_dot.m
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


class TestSpherefunvDot:
    def test_all_matlab_assertions(self):
        # pass(1): empty inputs give an empty result.
        assert Spherefunv.empty().dot(Spherefunv.empty()) is None \
            or Spherefunv.empty().isempty()
        # pass(2): a gradient field is tangent to the sphere.
        f = _sph(lambda x, y, z: jnp.cos((x + 0.1) * y * z))
        u = f.gradient()
        nrml = Spherefunv(_sph(lambda x, y, z: x),
                          _sph(lambda x, y, z: y),
                          _sph(lambda x, y, z: z))
        assert _norm_inf(u.dot(nrml)) < 1e4 * TOL
        # pass(3)/(4): dot = sum of componentwise products.
        u = Spherefunv(_sph(lambda x, y, z: x * z * jnp.cos(2 * y)),
                       _sph(lambda x, y, z: y * z * jnp.sin(2 * x)),
                       _sph(lambda x, y, z: jnp.exp(x * y * z)))
        v = Spherefunv(_sph(lambda x, y, z: x * y),
                       _sph(lambda x, y, z: y * z),
                       _sph(lambda x, y, z: z * x))
        f2 = u.dot(v)
        gsum = (u.components[0] * v.components[0]
                + u.components[1] * v.components[1]
                + u.components[2] * v.components[2])
        assert _norm_inf(f2 - gsum) < 10 * TOL
