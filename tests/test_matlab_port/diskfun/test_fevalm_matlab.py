"""Port of MATLAB Chebfun tests/diskfun/test_fevalm.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_fevalm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 100 * 1e3 * _EPS  # 100 * chebfun2eps-scale tolerance (as in MATLAB test)


class TestDiskfunFevalm:
    def test_empty_diskfun(self):
        # MATLAB: fevalm of an empty diskfun is empty.
        f = Diskfun.empty()
        t = np.pi * (2 * np.random.rand(5) - 1)
        r = 2 * np.random.rand(5) - 1
        B = f.fevalm(t, r)
        assert B.size == 0

    def test_rank_one(self):
        # f(theta, r) = r sin(theta)
        f = Diskfun.from_function(lambda t, r: r * jnp.sin(t))
        rng = np.random.default_rng(2016)
        t = np.pi * (2 * rng.random(5) - 1)
        r = rng.random(5)
        tt, rr = np.meshgrid(t, r)
        A = np.asarray(f(jnp.asarray(tt), jnp.asarray(rr)))
        B = np.asarray(f.fevalm(t, r))
        assert A.shape == (len(r), len(t))
        assert np.linalg.norm(A - B) < _TOL

    def test_essentially_one_dim(self):
        # f(theta, r) = exp(-r^2)
        f = Diskfun.from_function(lambda t, r: jnp.exp(-r**2))
        rng = np.random.default_rng(17)
        t = np.pi * (2 * rng.random(5) - 1)
        r = rng.random(5)
        tt, rr = np.meshgrid(t, r)
        A = np.asarray(f(jnp.asarray(tt), jnp.asarray(rr)))
        B = np.asarray(f.fevalm(t, r))
        assert np.linalg.norm(A - B) < _TOL
