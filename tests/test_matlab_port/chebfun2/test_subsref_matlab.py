"""Port of MATLAB Chebfun tests/chebfun2/test_subsref.m (Fable 5).

MATLAB subsref forms map to ``f(x, y)`` evaluation, ``f(":", y0)`` /
``f(x0, ":")`` slices, ``f(c1, c2)`` compositions and ``f.restrict``
(the ``f{a, b, c, d}`` form).

Provenance
----------
MATLAB source : tests/chebfun2/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _n(f):
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


class TestChebfun2Subsref:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = chebfun2(lambda x, y: jnp.sin(x * (y - .1)), domain=(-2, 2, -3, 4))
        assert abs(float(f(pi / 4, pi / 6)) - np.sin(pi / 4 * (pi / 6 - .1))) < tol  # pass(1)
        slice1 = chebfun(lambda x: jnp.sin(x * (pi / 6 - .1)), domain=(-2.0, 2.0))
        slice2 = chebfun(lambda x: jnp.sin(x * (pi / 4 - .1)), domain=(-2.0, 2.0))
        both = chebfun(lambda x: jnp.stack([slice1(x), slice2(x)], axis=-1),
                       domain=(-2.0, 2.0)).T
        assert _n(f(":", [pi / 6, pi / 4]) - both) < tol                 # pass(2)
        slice1 = chebfun(lambda y: jnp.sin(pi / 4 * (y - .1)), domain=(-3.0, 4.0))
        slice2 = chebfun(lambda y: jnp.sin(pi / 6 * (y - .1)), domain=(-3.0, 4.0))
        assert _n(f(pi / 4, ":") - slice1) < tol                         # pass(3)
        both = chebfun(lambda y: jnp.stack([slice1(y), slice2(y)], axis=-1),
                       domain=(-3.0, 4.0))
        assert _n(f([pi / 4, pi / 6], ":") - both) < tol                 # pass(4)
        assert float((f(":", ":") - f).norm()) < tol                    # pass(5)

        f = chebfun2(lambda x, y: x * y)
        c1 = chebfun(lambda t: 1 + 0 * t)
        c2 = chebfun(lambda t: -.3 + 0 * t)
        assert _n(f(c1, c2) + .3) < tol                                  # pass(6)
        assert _n(f(c1 + 1j * c2) + .3) < tol                            # pass(7)
        both = chebfun(lambda t: jnp.stack([c1(t), c2(t)], axis=-1))
        assert _n(f(both) - f(c1, c2)) < tol                             # pass(8): compose([c1, c2], f)

        f = chebfun2(lambda x, y: x)
        assert _n(f.rows.cols[0] - chebfun(lambda x: x)) < tol           # pass(9)
        assert np.linalg.norm(np.asarray(f.domain) - [-1, 1, -1, 1]) < tol  # pass(10)

        f = chebfun2(lambda x, y: jnp.sin(x * (y - .1)), domain=(-2, 2, -3, 4))
        g = f.restrict((-1, 1, -.5, .25))                                 # f{-1,1,-.5,.25}
        exact = chebfun2(lambda x, y: jnp.sin(x * (y - .1)), domain=(-1, 1, -.5, .25))
        assert float((g - exact).norm()) < 10 * tol                      # pass(11)

        f = chebfun2(lambda z: z)
        assert abs(complex(f(1j)) - 1j) < tol                            # pass(12)
        assert abs(complex(f(1)) - 1) < tol                              # pass(13)

        f = chebfun2(lambda x, y: jnp.cos(x * y))
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y) if hasattr(
            Chebfun2v, "from_functions") else Chebfun2v(
            [chebfun2(lambda x, y: x).approx, chebfun2(lambda x, y: y).approx])
        assert float((f(F) - f).norm()) < tol                            # pass(14)
        f = chebfun2(lambda x, y: jnp.cos(x * y), domain=(-3, 4, -2, 6))
        g = chebfun2(lambda x, y: jnp.cos(x * y), domain=(-2, 2, -2, 2))
        F = Chebfun2v([chebfun2(lambda x, y: x, domain=(-2, 2, -2, 2)).approx,
                       chebfun2(lambda x, y: y, domain=(-2, 2, -2, 2)).approx])
        assert float((f(F) - g).norm()) < tol                            # pass(15)

        F3 = Chebfun3v([Chebfun3.from_function(lambda x, y, z: x),
                        Chebfun3.from_function(lambda x, y, z: y)])
        g = chebfun2(lambda x, y: x + y)
        h = g(F3)
        h_true = Chebfun3.from_function(lambda x, y, z: x + y)
        assert float((h - h_true).norm()) < tol                          # pass(16)

        f = chebfun2(lambda x, y: x + 1j * y)
        g = chebfun2(lambda x, y: x + y)
        assert float((g(f) - g).norm()) < tol                            # pass(17)

        f3 = Chebfun3.from_function(lambda x, y, z: x + 1j * y)
        h = g(f3)
        h_true = Chebfun3.from_function(lambda x, y, z: x + y)
        assert float((h - h_true).norm()) < tol                          # pass(18)
        f3 = Chebfun3.from_function(lambda x, y, z: x)
        h = g(f3)
        h_true = Chebfun3.from_function(lambda x, y, z: x)
        assert float((h - h_true).norm()) < tol                          # pass(19)
        f1 = Chebfun3.from_function(lambda x, y, z: x)
        f2 = Chebfun3.from_function(lambda x, y, z: y)
        h = g(f1, f2)
        h_true = Chebfun3.from_function(lambda x, y, z: x + y)
        assert float((h - h_true).norm()) < tol                          # pass(20)

        f1 = chebfun2(lambda x, y: x)
        f2 = chebfun2(lambda x, y: y)
        h = g(f1, f2)
        h_true = chebfun2(lambda x, y: x + y)
        assert float((h - h_true).norm()) < tol                          # pass(21)

        F = chebfun(lambda t: jnp.stack([t, t], axis=-1))
        h = g(F)
        h_true = chebfun(lambda t: 2 * t)
        assert _n(h - h_true) < tol                                      # pass(22)
        f = chebfun(lambda t: t)
        h = g(f, f)
        assert _n(h - h_true) < tol                                      # pass(23)
