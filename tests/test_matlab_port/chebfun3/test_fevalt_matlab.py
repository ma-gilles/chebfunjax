"""Port of MATLAB Chebfun tests/chebfun3/test_fevalt.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.fevalt()`` now
evaluates on the tensor grid formed by its three coordinate vectors
(as opposed to ``f(x, y, z)``, which broadcasts pointwise).

MATLAB pass(3) builds a 'trig' Chebfun3; the trigonometric tech option
is not wired into the constructor, so that case is ported without the
flag -- fevalt's behaviour does not depend on the underlying tech.

Provenance
----------
MATLAB source : tests/chebfun3/test_fevalt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.utils.quadrature import chebpts_ab

EPS = float(np.finfo(np.float64).eps)
TOL = 1e4 * EPS


def _ff(x, y, z):
    return jnp.cos(x) + jnp.sin(x * y) + jnp.sin(z * x)


def _ff_np(x, y, z):
    return np.cos(x) + np.sin(x * y) + np.sin(z * x)


def _check(f, x, y, z):
    F = np.asarray(f.fevalt(x, y, z))
    xx, yy, zz = np.meshgrid(np.asarray(x), np.asarray(y), np.asarray(z),
                             indexing="ij")
    exact = _ff_np(xx, yy, zz)
    assert F.shape == exact.shape
    return float(np.linalg.norm((exact - F).ravel()))


class TestChebfun3Fevalt:
    def test_random_points_default_domain(self):
        # pass(1): a 10x10x10 tensor grid of random points.
        f = chebfun3(_ff)
        rng = np.random.default_rng(42)
        x, y, z = (rng.random(10) for _ in range(3))
        assert _check(f, x, y, z) < TOL * f.vscale()

    def test_chebyshev_points_on_a_box(self):
        # pass(2): 20 Chebyshev points per direction on [-1,1] x
        # [-pi/3,pi/3] x [-2,0].
        dom = (-1.0, 1.0, -np.pi / 3, np.pi / 3, -2.0, 0.0)
        f = chebfun3(_ff, domain=dom)
        x = np.asarray(chebpts_ab(20, dom[0], dom[1], kind=2))
        y = np.asarray(chebpts_ab(20, dom[2], dom[3], kind=2))
        z = np.asarray(chebpts_ab(20, dom[4], dom[5], kind=2))
        assert _check(f, x, y, z) < TOL * f.vscale()

    def test_equispaced_vector_inputs(self):
        # pass(3): 20 equispaced points per direction.
        f = chebfun3(_ff)
        x = np.linspace(-1.0, 1.0, 20)
        assert _check(f, x, x, x) < TOL * f.vscale()

    def test_fevalt_matches_pointwise_eval_on_the_diagonal(self):
        # fevalt's diagonal entries are the pointwise evaluations.
        f = chebfun3(_ff)
        pts = np.linspace(-0.9, 0.9, 5)
        F = np.asarray(f.fevalt(pts, pts, pts))
        diag = np.array([F[i, i, i] for i in range(len(pts))])
        direct = np.asarray(f(jnp.asarray(pts), jnp.asarray(pts),
                              jnp.asarray(pts)))
        assert float(np.max(np.abs(diag - direct))) < TOL
