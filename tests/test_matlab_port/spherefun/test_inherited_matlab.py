"""Port of MATLAB Chebfun tests/spherefun/test_inherited.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_inherited.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

from ._cart import sph_lt, sph_xyz

jax.config.update("jax_enable_x64", True)

D2R = np.pi / 180.0


def _n(f):
    return float(f.norm())


class TestSpherefunInherited:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = sph_xyz(lambda x, y, z: -1 + 0 * x)
        h = sph_xyz(lambda x, y, z: 1 + 0 * x)
        assert _n(abs(f) - h) < tol                                          # pass(1)
        f = sph_xyz(lambda x, y, z: jnp.cos(x))
        assert _n(f.cos() - sph_xyz(lambda x, y, z: jnp.cos(jnp.cos(x)))) < tol  # pass(2)
        assert _n(f.cosh() - sph_xyz(lambda x, y, z: jnp.cosh(jnp.cos(x)))) < tol  # pass(3)
        f = sph_xyz(lambda x, y, z: x)
        assert _n(f - f.conj()) < tol                                        # pass(4)
        f = sph_xyz(lambda x, y, z: y)
        assert _n(f - f.ctranspose()) < tol                                  # pass(5)
        f = sph_lt(lambda lam, th: jnp.sin(th) * jnp.cos(lam))
        h = chebfun(lambda x: jnp.sin(x) * jnp.cos(x), domain=(0.0, np.pi))
        assert float((f.diag() - h).norm(2)) < tol                           # pass(6)
        f = sph_xyz(lambda x, y, z: jnp.cos(x) + jnp.sin(y))
        assert _n(f.exp() - sph_xyz(lambda x, y, z: jnp.exp(jnp.cos(x) + jnp.sin(y)))) < tol  # pass(7)

        f = sph_lt(lambda lam, th: jnp.sin(th) * jnp.cos(lam))
        g1 = sph_lt(lambda lam, th: jnp.sin(th) * jnp.cos(-lam))
        g2 = sph_lt(lambda lam, th: jnp.sin(-th) * jnp.cos(lam))
        assert _n(g1 - f.fliplr()) < tol                                     # pass(8)
        assert _n(g2 - f.flipud()) < tol                                     # pass(9)
        assert _n(g2 - f.flipdim(1)) + _n(g1 - f.flipdim(2)) < tol           # pass(10)

        f = sph_xyz(lambda x, y, z: x)
        assert _n(f.imag()) < tol                                            # pass(11)
        f = sph_xyz(lambda x, y, z: jnp.cos(x))
        g = sph_xyz(lambda x, y, z: jnp.cos(-x))
        assert f.isequal(g)                                                  # pass(12)
        g = sph_xyz(lambda x, y, z: jnp.cos(x) + 1)
        assert not f.isequal(g)                                              # pass(13)
        assert sph_xyz(lambda x, y, z: x).isreal()                           # pass(14)
        assert sph_xyz(lambda x, y, z: jnp.cos(x * y * z)).isreal()          # pass(15)
        assert sph_xyz(lambda x, y, z: 0 * jnp.cos(x * y * z)).iszero()      # pass(16)
        assert Spherefun.from_values(np.zeros((10, 10))).iszero()            # pass(17)
        f = sph_xyz(lambda x, y, z: jnp.cos(z))
        m, _n2 = f.length()
        assert m == 1                                                        # pass(18)
        f = sph_xyz(lambda x, y, z: jnp.exp(z))
        assert _n(f.log() - sph_xyz(lambda x, y, z: z)) < tol                # pass(19)
        f = sph_xyz(lambda x, y, z: jnp.cos(x * y))
        assert _n(f.sin() - sph_xyz(lambda x, y, z: jnp.sin(jnp.cos(x * y)))) < tol  # pass(20)
        assert _n(f.sinh() - sph_xyz(lambda x, y, z: jnp.sinh(jnp.cos(x * y)))) < tol  # pass(21)
        assert f.size(1) == np.inf and f.size(2) == np.inf                   # pass(22)
        f2 = sph_xyz(lambda x, y, z: jnp.cos(x * y) ** 2)
        assert _n(f2.sqrt() - sph_xyz(lambda x, y, z: jnp.cos(x * y))) < tol  # pass(23)
        assert _n(f.tan() - sph_xyz(lambda x, y, z: jnp.tan(jnp.cos(x * y)))) < 10 * tol  # pass(24)
        assert _n(f.tand() - sph_xyz(lambda x, y, z: jnp.tan(D2R * jnp.cos(x * y)))) < tol  # pass(25)
        assert _n(f.tanh() - sph_xyz(lambda x, y, z: jnp.tanh(jnp.cos(x * y)))) < tol  # pass(26)
        assert _n(f.uminus() - sph_xyz(lambda x, y, z: -jnp.cos(x * y))) < tol  # pass(27)
        assert _n(f.uplus() - sph_xyz(lambda x, y, z: +jnp.cos(x * y))) < tol  # pass(28)
