"""Port of MATLAB Chebfun tests/spherefun/test_sample.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_sample.m
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


class TestSpherefunSample:
    def test_all_matlab_assertions(self):
        tol = 100 * 100 * TOL2
        f = _sph(lambda x, y, z: jnp.sin(jnp.pi * x * y))
        m, n = f.length()
        nn, mm = np.asarray(f.sample()).shape
        assert (m == mm) and (n == nn)                        # pass(1)
        nn, mm = np.asarray(f.sample(120, 121)).shape
        assert (120 == mm) and (121 == nn)                    # pass(2)
        nn, mm = np.asarray(f.sample(121, 120)).shape
        assert (121 == mm) and (120 == nn)                    # pass(3)
        m, n = 30, 20
        from chebfunjax.utils.quadrature import trigpts
        lam = np.pi * np.array(trigpts(m)[0])
        th = np.linspace(0.0, np.pi, n)
        L, T = np.meshgrid(lam, th)
        F = np.asarray(f(jnp.asarray(L), jnp.asarray(T)))
        G = np.asarray(f.sample(m, n))
        assert np.max(np.abs(F - G)) < tol                    # pass(4)
        U, D, V = f.sample_cdr(m, n)
        G2 = np.asarray(U) @ np.asarray(D) @ np.asarray(V).T
        assert np.max(np.abs(G2 - G)) < tol                   # pass(5)
