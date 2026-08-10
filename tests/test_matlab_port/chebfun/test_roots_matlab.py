"""Port of MATLAB Chebfun tests/chebfun/test_roots.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunRoots:
    def test_thousand_roots(self):
        M = 1000
        f = cj.chebfun(lambda x: jnp.sin(M * np.pi * x),
                       domain=(0.0, 1.0))
        r = np.sort(np.asarray(f.roots()))
        exact = np.linspace(0, 1, M + 1)
        assert len(r) == M + 1
        assert float(np.max(np.abs(r - exact))) < 1e2 * EPS

    def test_quartic_with_tiny_perturbation(self):
        f = cj.chebfun(lambda x: (x - 0.1) * (x + 0.9) * x * (x - 0.9)
                       + 1e-14 * x ** 5)
        r = np.asarray(f.roots())
        assert len(r) == 4
        vals = np.asarray(f(jnp.asarray(r)))
        assert float(np.max(np.abs(vals))) < 1e3 * EPS

    def test_jump_root_options(self):
        # MATLAB pass(3, 4): 'nojump' suppresses sign changes across a
        # jump and 'nozerofun' suppresses the midpoint root of an
        # identically-zero piece.
        Fs = cj.chebfun([-1.0, 2.0], domain=[-2.0, 0.0, 1.0])
        Fh = cj.chebfun([-1.0, 0.0, 1.0], domain=[-2.0, -1.0, 0.0, 2.0])
        assert len(np.asarray(Fs.roots(nojump=True))) == 0
        assert len(np.asarray(Fh.roots(nojump=True, nozerofun=True))) == 0

        rs = np.asarray(Fs.roots())
        assert len(rs) == 1 and rs[0] == 0.0
        rh = np.asarray(Fh.roots(nozerofun=True))
        assert len(rh) == 2 and rh[0] == -1.0 and rh[1] == 0.0
        # Without 'nozerofun' the zero piece contributes its midpoint.
        rh_all = np.asarray(Fh.roots())
        assert -0.5 in rh_all
