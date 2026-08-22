"""Port of MATLAB Chebfun tests/spherefun/test_fevalm.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_fevalm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

TOL2 = 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


class TestSpherefunFevalm:
    def test_all_matlab_assertions(self):
        tol = 100 * 100 * TOL2
        rng = np.random.RandomState(2016)
        # pass(1): empty
        B = Spherefun.empty().fevalm(rng.rand(5), rng.rand(5))
        assert np.asarray(B).size == 0
        # pass(3)-style: spherefun fevalm == meshgrid feval
        f = _sph(lambda x, y, z: jnp.exp(-(z - 1.0) ** 2))
        s = np.pi * (2 * rng.rand(5) - 1)
        t = np.pi / 2 * rng.rand(5)
        ss, tt = np.meshgrid(s, t)
        A = np.asarray(f(jnp.asarray(ss), jnp.asarray(tt)))
        B = np.asarray(f.fevalm(jnp.asarray(s), jnp.asarray(t)))
        assert np.linalg.norm(A - B) < tol
