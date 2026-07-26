"""Port of MATLAB Chebfun tests/spherefun/test_rotate.m (Fable 5).

FIXED: Spherefun.rotate added in the Fable 5 audit (ZYZ Euler,
feval-based).  The 'nufft' fast path is not implemented; the feval
path is the reference method in MATLAB too.

Provenance
----------
MATLAB source : tests/spherefun/test_rotate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e3 * np.finfo(float).eps
ANGLS = (0.704147450733711, 1.158707918472494, 0.884902035498222)

LAMS = jnp.asarray(np.linspace(-3.0, 3.0, 13))
THS = jnp.asarray(np.linspace(0.05, 3.09, 13))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")


def _maxdiff(f, g):
    return float(jnp.max(jnp.abs(f(LL, TT) - g(LL, TT))))


class TestSpherefunRotate:
    def test_rotate_and_undo(self):
        # MATLAB pass(3): a rotate-and-undo round-trip returns f to within the
        # 10*TOL bound.  Reached via the fastSphereEval port (2D-NUFFT
        # ~1e-15/point evaluation) plus fixed-bandlimit resampling in
        # Spherefun.rotate; the round-trip lands at ~9.6e-13 (margin ~2.3x).
        f = Spherefun.from_function(
            lambda lam, th: jnp.sin(
                jnp.cos(th) + jnp.cos(lam - 0.2) * jnp.sin(th)
                + jnp.sin(lam + 0.4) * jnp.sin(th)) ** 8)
        g = f.rotate(*ANGLS)
        h = g.rotate(-ANGLS[2], -ANGLS[1], -ANGLS[0])
        assert _maxdiff(h, f) < 10 * TOL

    def test_rotation_preserves_integral(self):
        # pass(4): integral preserved under rotation
        f = Spherefun.from_function(
            lambda lam, th: jnp.sin(
                jnp.cos(th) + jnp.cos(lam - 0.2) * jnp.sin(th)
                + jnp.sin(lam + 0.4) * jnp.sin(th)) ** 8)
        g = f.rotate(*ANGLS)
        assert abs(float(g.sum2()) - float(f.sum2())) < TOL

    def test_z_symmetric_invariant_under_z_rotation(self):
        # pass(5): a function of z only is invariant under Rz
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(th) ** 3
            - 0.6 * jnp.cos(th))
        g = f.rotate(np.sqrt(2.0), 0.0, 0.0)
        assert _maxdiff(g, f) < TOL
