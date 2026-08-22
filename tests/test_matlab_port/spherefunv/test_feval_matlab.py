"""Port of MATLAB Chebfun tests/spherefunv/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_feval.m
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




class TestSpherefunvFeval:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(9)
        f = lambda x, y, z: jnp.sin(x + y * z) + 1
        g = lambda x, y, z: jnp.cos(x + y * z) + 1
        h = lambda x, y, z: z * (x + y * z) + 1
        u = Spherefunv(_sph(f), _sph(g), _sph(h))

        # pass(1)/(2): cartesian points on the sphere (converted to
        # spherical for the chebfunjax call convention).
        for npts in (1, 10):
            P = rng.rand(npts, 3)
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
            lam = jnp.asarray(np.arctan2(P[:, 1], P[:, 0]))
            th = jnp.asarray(np.arccos(P[:, 2]))
            vals = u(lam, th)
            X = jnp.asarray(P[:, 0])
            Y = jnp.asarray(P[:, 1])
            Z = jnp.asarray(P[:, 2])
            want = [f(X, Y, Z), g(X, Y, Z), h(X, Y, Z)]
            for got, w in zip(vals, want):
                assert float(jnp.max(jnp.abs(
                    jnp.asarray(got) - w))) < TOL

        # pass(3)/(4): spherical-coordinate evaluation.
        for npts in (1, 10):
            lam = jnp.asarray(rng.rand(npts))
            th = jnp.asarray(rng.rand(npts))
            X = jnp.cos(lam) * jnp.sin(th)
            Y = jnp.sin(lam) * jnp.sin(th)
            Z = jnp.cos(th)
            vals = u(lam, th)
            want = [f(X, Y, Z), g(X, Y, Z), h(X, Y, Z)]
            for got, w in zip(vals, want):
                assert float(jnp.max(jnp.abs(
                    jnp.asarray(got) - w))) < TOL
