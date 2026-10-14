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

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)

TOL = 10 * ChebfunPref().cheb2Prefs.chebfun2eps


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _sampled_norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_sampled_norm_inf(c, n) for c in F.components)


class TestSpherefunvDot:
    def test_all_matlab_assertions(self):
        # pass(1): empty inputs give an empty result.
        h = Spherefunv.empty().dot(Spherefunv.empty())
        assert isinstance(h, Spherefun) and h.isempty()
        # pass(2): a gradient field is tangent to the sphere.
        f = _sph(lambda x, y, z: jnp.cos((x + 0.1) * y * z))
        u = f.grad()
        nrml = Spherefunv(_sph(lambda x, y, z: x),
                          _sph(lambda x, y, z: y),
                          _sph(lambda x, y, z: z))
        assert u.dot(nrml).norm(jnp.inf) < TOL
        # pass(3)/(4): dot() and transpose-times agree with the component sum.
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
        assert (gsum - f2).norm() < TOL
        g = u.ctranspose() @ v
        assert (g - f2).norm() < TOL

    def test_supplemental_sampled_dot_control(self):
        # Preserve the earlier sampled diagnostic separately from the four
        # native source predicates above.
        f = _sph(lambda x, y, z: jnp.cos((x + 0.1) * y * z))
        u = f.grad()
        nrml = Spherefunv(_sph(lambda x, y, z: x),
                          _sph(lambda x, y, z: y),
                          _sph(lambda x, y, z: z))
        assert _sampled_norm_inf(u.dot(nrml)) < 1e4 * TOL
