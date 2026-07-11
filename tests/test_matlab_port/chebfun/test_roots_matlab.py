"""Port of MATLAB Chebfun tests/chebfun/test_roots.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

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
        pytest.skip("chebfunjax roots has no 'nojump'/'nozerofun' options")
