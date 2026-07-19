"""Port of MATLAB Chebfun tests/chebfun2/test_fevalm.m (Fable 5).

FIXED: Chebfun2.fevalm (tensor-grid evaluation) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_fevalm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Fevalm:
    def test_empty(self):
        f = Chebfun2.empty()
        s = 2 * np.random.default_rng(0).random(5) - 1
        t = 2 * np.random.default_rng(1).random(5) - 1
        B = np.asarray(f.fevalm(s, t))
        assert B.size == 0

    def test_symmetric_function(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        rng = np.random.default_rng(2016)
        s = 2 * rng.random(5) - 1
        t = 2 * rng.random(5) - 1
        ss, tt = np.meshgrid(s, t)
        A = np.asarray(f(jnp.asarray(ss), jnp.asarray(tt)))
        B = np.asarray(f.fevalm(s, t))
        assert np.linalg.norm(A - B) < TOL

    def test_essentially_one_dimensional(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(y - 0.1))
        rng = np.random.default_rng(7)
        s = 2 * rng.random(5) - 1
        t = 2 * rng.random(5) - 1
        ss, tt = np.meshgrid(s, t)
        A = np.asarray(f(jnp.asarray(ss), jnp.asarray(tt)))
        B = np.asarray(f.fevalm(s, t))
        assert np.linalg.norm(A - B) < TOL

    def test_complex_valued(self):
        # MATLAB: chebfun2(@(z) cos(z)) with z = x + 1i*y.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + 1j * y))
        rng = np.random.default_rng(11)
        s = 2 * rng.random(6) - 1
        t = 2 * rng.random(6) - 1
        ss, tt = np.meshgrid(s, t)
        A = np.asarray(f(jnp.asarray(ss), jnp.asarray(tt)))
        B = np.asarray(f.fevalm(s, t))
        assert np.linalg.norm(A - B) < TOL
