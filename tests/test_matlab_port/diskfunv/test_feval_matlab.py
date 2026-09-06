"""Port of MATLAB Chebfun tests/diskfunv/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfunv import Diskfunv

from ..diskfun._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunvFeval:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        rng = np.random.RandomState(9)
        f = lambda x, y: jnp.sin(x + y) + 1  # noqa: E731
        g = lambda x, y: jnp.cos(x + y) + 1  # noqa: E731
        u = Diskfunv(disk_xy(f), disk_xy(g))

        def uexact(x, y):
            return np.stack([np.asarray(f(jnp.asarray(x), jnp.asarray(y))).ravel(),
                             np.asarray(g(jnp.asarray(x), jnp.asarray(y))).ravel()])
        x = rng.rand(1, 2)
        x = x / np.sqrt(np.sum(x ** 2))
        y = x[0, 1]
        x = x[0, 0]
        assert np.max(np.abs(np.asarray(u.feval(x, y)) - uexact(x, y))) < tol  # pass(1)
        x = rng.rand(10, 2)
        nrm = np.sqrt(np.sum(x ** 2, axis=1))
        y = x[:, 1] / nrm
        x = x[:, 0] / nrm
        assert np.max(np.abs(np.asarray(u.feval(x, y)) - uexact(x, y))) < tol  # pass(2)

        def fp(th, r):
            return f(r * jnp.cos(th), r * jnp.sin(th))

        def gp(th, r):
            return g(r * jnp.cos(th), r * jnp.sin(th))

        def uexact_p(th, r):
            return np.stack([np.asarray(fp(jnp.asarray(th), jnp.asarray(r))).ravel(),
                             np.asarray(gp(jnp.asarray(th), jnp.asarray(r))).ravel()])
        th = 2 * rng.rand() - 1
        rad = np.pi * (2 * rng.rand() - 1)
        assert np.max(np.abs(np.asarray(u.feval(th, rad, "polar")) - uexact_p(th, rad))) < tol  # pass(3)
        th = np.pi * (2 * rng.rand(10) - 1)
        rad = 2 * rng.rand(10) - 1
        assert np.max(np.abs(np.asarray(u.feval(th, rad, "polar")) - uexact_p(th, rad))) < tol  # pass(4)
        with pytest.raises(ValueError, match="pointsNotOnDisk"):
            u.feval(1, 2)                                                    # pass(5)
