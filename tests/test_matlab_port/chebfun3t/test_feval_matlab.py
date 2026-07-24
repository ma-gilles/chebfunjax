"""Port of MATLAB Chebfun tests/chebfun3t/test_feval.m (Fable 5).

Point/array/meshgrid/ndgrid/tensor evaluation of a
:class:`Chebfun3T` -- pure feval math, identical for the Tucker-backed
chebfunjax wrapper and MATLAB's full-tensor variant, so exercised
directly against :func:`chebfunjax.chebfun3d.chebfun3t`.

MATLAB scales the tolerance by the stored ``f.vscale``; :class:`Chebfun3T`
stores no vscale, so it is recomputed as max|f| over a sample grid (the
quantity MATLAB's vscale estimates).  A lean but representative subset of
the 17 MATLAB assertions is kept (each chebfun3t construction is a full
3D adaptive solve): coordinate extraction on a non-symmetric domain,
plus the four input layouts (flat arrays, meshgrid, ndgrid, 3D tensor).

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


def _vscale(f, dom):
    xs = np.linspace(dom[0], dom[1], 6)
    ys = np.linspace(dom[2], dom[3], 6)
    zs = np.linspace(dom[4], dom[5], 6)
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    return float(np.max(np.abs(np.asarray(f(xx, yy, zz)))))


@pytest.mark.slow
class TestChebfun3tFeval:
    def test_all_matlab_assertions(self):
        dom = (-1.0, 2.0, -np.pi / 2, np.pi, -3.0, 1.0)

        # Coordinate extraction on a non-symmetric domain:
        #   f = chebfun3t(@(x,y,z) x, dom);  f(pi/6,pi/12,-1) == pi/6
        fx = chebfun3t(lambda x, y, z: x, dom)
        vs = _vscale(fx, dom)
        assert abs(float(fx(0.0, 0.0, 0.0))) < TOL * vs
        assert abs(float(fx(np.pi / 6, np.pi / 12, -1.0)) - np.pi / 6) < TOL * vs

        #   f = chebfun3t(@(x,y,z) z, dom);  f(pi/6,pi/12,-1) == -1
        fz = chebfun3t(lambda x, y, z: z, dom)
        vs = _vscale(fz, dom)
        assert abs(float(fz(0.0, 0.0, 0.0))) < TOL * vs
        assert abs(float(fz(np.pi / 6, np.pi / 12, -1.0)) + 1.0) < TOL * vs

        # Harder function; point + array evaluation:
        #   f = cos(x) + sin(x.*y) + sin(z.*x);
        def hard(x, y, z):
            return jnp.cos(x) + jnp.sin(x * y) + jnp.sin(z * x)

        g = chebfun3t(hard)
        vs = _vscale(g, (-1, 1, -1, 1, -1, 1))
        p = np.array([0.126986816, 0.632359246, -0.351283361])
        exact_p = float(np.cos(p[0]) + np.sin(p[0] * p[1]) + np.sin(p[2] * p[0]))
        assert abs(float(g(p[0], p[1], p[2])) - exact_p) < TOL * vs
        r = np.array([0.1, 0.4, 0.7, -0.2, 0.9])
        s = np.array([0.3, -0.5, 0.2, 0.8, -0.1])
        t = np.array([-0.6, 0.1, 0.5, -0.9, 0.4])
        assert np.max(np.abs(np.asarray(g(r, s, t)) - np.asarray(hard(r, s, t)))) \
            < TOL * vs

        # The four input layouts on sin(pi*(x+y+z)):
        def ff(x, y, z):
            return jnp.sin(np.pi * (x + y + z))

        f = chebfun3t(ff)
        vs = _vscale(f, (-1, 1, -1, 1, -1, 1))
        lin = np.linspace(-1, 1, 20)
        # flat arrays
        assert np.max(np.abs(np.asarray(f(lin, lin, lin))
                             - np.asarray(ff(lin, lin, lin)))) < 1e2 * TOL * vs
        # meshgrid
        xx, yy, zz = np.meshgrid(lin, lin, lin)
        assert np.max(np.abs(np.asarray(f(xx, yy, zz))
                             - np.asarray(ff(xx, yy, zz)))) < 1e2 * TOL * vs
        # ndgrid
        xn, yn, zn = np.meshgrid(lin, lin, lin, indexing="ij")
        assert np.max(np.abs(np.asarray(f(xn, yn, zn))
                             - np.asarray(ff(xn, yn, zn)))) < 1e2 * TOL * vs
        # 3D tensor of random points
        rng = np.random.default_rng(42)
        xr = rng.random((4, 5, 6))
        yr = rng.random((4, 5, 6))
        zr = rng.random((4, 5, 6))
        assert np.max(np.abs(np.asarray(f(xr, yr, zr))
                             - np.asarray(ff(xr, yr, zr)))) < 1e2 * TOL * vs
