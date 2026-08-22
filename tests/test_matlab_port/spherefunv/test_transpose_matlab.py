"""Port of MATLAB Chebfun tests/spherefunv/test_transpose.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_transpose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)




class TestSpherefunvTranspose:
    def test_all_matlab_assertions(self):
        e = Spherefunv.empty()
        assert e.transpose().isempty()  # pass(1)/(2)
        assert e.ctranspose().isempty()

        f = _sph(lambda x, y, z: jnp.cos((x + 0.1) * y * z))
        u = f.gradient()
        m, n, p = u.ctranspose().size
        assert np.isinf(m) and np.isinf(n) and p == 3  # pass(3)
        m, n, p = u.transpose().size
        assert np.isinf(m) and np.isinf(n) and p == 3  # pass(4)
        # pass(5)/(6): u' - u.' vanishes for real fields; double
        # transpose is the identity.
        w = [a - b for a, b in zip(u.ctranspose().components,
                                   u.transpose().components)]
        assert max(_norm_inf(c) for c in w) < TOL
        v = u.transpose()
        assert _vnorm(Spherefunv(*[
            a - b for a, b in zip(u.components,
                                  v.transpose().components)])) < TOL
        # pass(7): u' * u is a spherefun (the dot contraction).
        from chebfunjax.spherefun.spherefun import Spherefun
        s = u.ctranspose() @ u
        assert isinstance(s, Spherefun)
