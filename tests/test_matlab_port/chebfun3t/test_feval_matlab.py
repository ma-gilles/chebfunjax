"""Literal port of all 17 MATLAB tests/chebfun3t/test_feval.m predicates.

The source's functions, domains, vector lengths, meshgrid/ndgrid dimensions,
random tensor shape, and tolerance multipliers are retained. ``vscale`` is the
stored public field of native Chebfun3T.

Provenance
----------
MATLAB source : tests/chebfun3t/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e2 * EPS
_DOMAIN = (-1.0, 2.0, -np.pi / 2, np.pi, -3.0, 1.0)


@pytest.mark.slow
class TestChebfun3tFeval:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(42)

        f = chebfun3t(lambda x, y, z: x, _DOMAIN)
        assert abs(float(f(0, 0, 0))) < TOL * float(f.vscale)
        assert abs(float(f(np.pi/6, np.pi/12, -1)) - np.pi/6) < TOL * float(f.vscale)

        f = chebfun3t(lambda x, y, z: y, _DOMAIN)
        assert abs(float(f(0, 0, 0)) - 0.0) < TOL
        assert abs(float(f(np.pi/6, np.pi/12, -1)) - np.pi/12) < TOL * float(f.vscale)

        f = chebfun3t(lambda x, y, z: z, _DOMAIN)
        assert abs(float(f(0, 0, 0)) - 0.0) < TOL
        assert abs(float(f(np.pi/6, np.pi/12, -1)) + 1.0) < TOL * float(f.vscale)

        def hard(x, y, z):
            return jnp.cos(x) + jnp.sin(x*y) + jnp.sin(z*x)

        g = chebfun3t(hard)
        pts = 2 * rng.rand(3, 1) - 1
        assert abs(float(hard(*pts[:, 0])) - float(g(*pts[:, 0]))) < TOL * float(g.vscale)

        r = rng.rand(10, 1)
        s = rng.rand(10, 1)
        t = rng.rand(10, 1)
        rr, ss, tt = np.meshgrid(r[:, 0], s[:, 0], t[:, 0])
        assert np.max(np.abs(np.asarray(hard(r[:, 0], s[:, 0], t[:, 0]))
                             - np.asarray(g(r[:, 0], s[:, 0], t[:, 0])))) < TOL * float(g.vscale)
        assert np.max(np.abs(np.asarray(hard(rr, ss, tt))
                             - np.asarray(g(rr, ss, tt)))) < TOL * float(g.vscale)

        g = chebfun3t(hard, (-np.pi/6, np.pi/2, -np.pi/12, np.sqrt(3), -3, 1))
        r0, s0, t0 = 0.126986816293506, 0.632359246225410, 0.351283361405006
        assert abs(float(hard(r0, s0, t0)) - float(g(r0, s0, t0))) < TOL * float(g.vscale)
        assert abs(float(hard(r0, s0, t0)) - float(g(r0, s0, t0))) < TOL * float(g.vscale)
        assert np.max(np.abs(np.asarray(hard(rr, ss, tt))
                             - np.asarray(g(rr, ss, tt)))) < TOL * float(g.vscale)

        ff = lambda x, y, z: jnp.sin(jnp.pi * (x + y + z))
        f = chebfun3t(ff)
        xx = np.linspace(-1, 1, 100)[:, None]
        yy = np.linspace(-1, 1, 100)[:, None]
        zz = np.linspace(-1, 1, 100)[:, None]
        assert np.linalg.norm(np.asarray(f(xx, yy, zz)).ravel()
                              - np.asarray(ff(xx, yy, zz)).ravel()) < 100 * TOL

        ff = lambda x, y, z: jnp.sin(jnp.pi * (x + y + z))
        f = chebfun3t(ff)
        xx, yy, zz = rng.rand(100, 1), rng.rand(100, 1), rng.rand(100, 1)
        assert np.linalg.norm(np.asarray(f(xx, yy, zz))
                              - np.asarray(ff(xx, yy, zz))) < 100 * TOL

        ff = lambda x, y, z: jnp.sin(jnp.pi * (x + y + z))
        f = chebfun3t(ff)
        grid = np.linspace(-1, 1, 100)
        xx, yy, zz = np.meshgrid(grid, grid, grid)
        assert np.linalg.norm(np.asarray(f(xx, yy, zz)).ravel()
                              - np.asarray(ff(xx, yy, zz)).ravel()) < 100 * TOL

        ff = lambda x, y, z: jnp.sin(jnp.pi * (x + y + z))
        f = chebfun3t(ff)
        xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
        assert np.linalg.norm(np.asarray(f(xx, yy, zz)).ravel()
                              - np.asarray(ff(xx, yy, zz)).ravel()) < 100 * TOL

        ff = lambda x, y, z: jnp.sin(jnp.pi * (x + y + z))
        f = chebfun3t(ff)
        xx, yy, zz = rng.rand(10, 20, 30), rng.rand(10, 20, 30), rng.rand(10, 20, 30)
        assert np.linalg.norm(np.asarray(f(xx, yy, zz)).ravel()
                              - np.asarray(ff(xx, yy, zz)).ravel()) < 100 * TOL
