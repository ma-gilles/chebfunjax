"""Port of MATLAB Chebfun tests/spherefunv/test_subsref.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_subsref.m
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



from chebfunjax.chebfun3d.chebfun3 import Chebfun3  # noqa: E402
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v  # noqa: E402


class TestSpherefunvSubsref:
    def test_all_matlab_assertions(self):
        # pass(1): F(1) recovers the first component with its pivots.
        f = _sph(lambda x, y, z: jnp.sin(10 * x * y * z))
        F = Spherefunv(f, f, f)
        G = F.components[0]
        assert np.allclose(np.asarray(G.pivots),
                           np.asarray(f.pivots))

        # pass(2)/(3): composition with chebfun3 / chebfun3v.
        F = Spherefunv(_sph(lambda x, y, z: x),
                       _sph(lambda x, y, z: y),
                       _sph(lambda x, y, z: z))
        g = Chebfun3.from_function(lambda x, y, z: x + y + z)
        h = F.compose(g)
        h_true = _sph(lambda x, y, z: x + y + z)
        assert _norm_inf(h - h_true) < TOL
        G = Chebfun3v([Chebfun3.from_function(lambda x, y, z: x),
                       Chebfun3.from_function(
                           lambda x, y, z: y + 0 * x),
                       Chebfun3.from_function(
                           lambda x, y, z: z + 0 * x)])
        H = F.compose(G)
        for c, ref in zip(H.components, F.components):
            assert _norm_inf(c - ref) < TOL
