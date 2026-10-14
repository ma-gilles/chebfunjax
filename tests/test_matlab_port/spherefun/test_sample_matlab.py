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
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


class TestSpherefunSample:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = _sph(lambda x, y, z: jnp.sin(jnp.pi * x * y))
        m, n = f.length()
        nn, mm = np.asarray(f.sample()).shape
        assert (m == mm) and (n == nn)                        # pass(1)
        nn, mm = np.asarray(f.sample(120, 121)).shape
        assert (120 == mm) and (121 == nn)                    # pass(2)
        nn, mm = np.asarray(f.sample(121, 120)).shape
        assert (121 == mm) and (120 == nn)                    # pass(3)
        from chebfunjax.utils.quadrature import trigpts

        def check_samples(m, n, pass_matrix, pass_cdr):
            lam = np.pi * np.asarray(trigpts(m)[0])
            th = np.linspace(0.0, np.pi, n)
            L, T = np.meshgrid(lam, th)
            F = np.asarray(f(jnp.asarray(L), jnp.asarray(T)))
            G = np.asarray(f.sample(m, n))
            assert np.max(np.abs(F.ravel() - G.ravel())) < tol, pass_matrix
            U, D, V = f.sample_cdr(m, n)
            G_cdr = np.asarray(U) @ np.asarray(D) @ np.asarray(V).T
            assert np.max(np.abs(F.ravel() - G_cdr.ravel())) < tol, pass_cdr

        check_samples(30, 20, "pass(4)", "pass(5)")
        check_samples(30, 21, "pass(6)", "pass(7)")
        check_samples(31, 20, "pass(8)", "pass(9)")
        check_samples(31, 21, "pass(10)", "pass(11)")

        f = _sph(lambda x, y, z: 1 + 0 * x)
        F = np.asarray(f.sample(128, 128))
        assert np.max(np.abs(F.ravel() - 1)) < tol              # pass(12)

        with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:sample:inputs"):
            f.sample(0, 20)                                     # pass(13)
        with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:sample:inputs"):
            f.sample(20, 0)                                     # pass(14)
