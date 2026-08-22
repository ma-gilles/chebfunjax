"""Port of MATLAB Chebfun tests/spherefun/test_composition_operators.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_composition_operators.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestSpherefunCompositionOperators:
    def test_all_matlab_assertions(self):
        tol = 1e3 * 100 * TOL2
        base = lambda x, y, z: jnp.cos(x * y) + jnp.sin(x * y) + z - 0.1
        f = _sph(base)
        x = _sph(lambda x, y, z: x)
        y = _sph(lambda x, y, z: y)
        z = _sph(lambda x, y, z: z)

        exact = _sph(lambda X, Y, Z: base(X, Y, Z)
                     * jnp.sin(Z * (X - 0.1) * (Y + 0.4)))
        prod = f * (z * (x - 0.1) * (y + 0.4)).sin()
        assert float((exact - prod).norm()) < 100 * tol          # pass(1)
        for op, name in ((jnp.cos, "cos"), (jnp.cosh, "cosh"),
                         (jnp.sin, "sin"), (jnp.sinh, "sinh")):
            exact = _sph(lambda X, Y, Z, o=op: o(base(X, Y, Z)))
            got = getattr(f, name)()
            assert float((exact - got).norm()) < tol             # 2-5

        f = _sph(lambda x, y, z: z + jnp.sin(jnp.pi * x * y))
        assert float((f + f + f - 3.0 * f).norm()) < 100 * tol   # pass(6)
        assert float((f * f - f ** 2).norm()) < tol              # pass(7)

        g = cj.chebfun(lambda t: t ** 2, domain=(-1.5, 1.5))
        h_true = _sph(lambda x, y, z: (z + jnp.sin(jnp.pi * x * y)) ** 2)
        h = f.compose(g)
        assert float((h - h_true).norm()) < tol                  # pass(8)

        G = [cj.chebfun(lambda t: t ** 2, domain=(-1.5, 1.5)),
             cj.chebfun(lambda t: t, domain=(-1.5, 1.5)),
             cj.chebfun(lambda t: -t ** 2, domain=(-1.5, 1.5))]
        H = f.compose(G)
        H_true = [h_true, f, -1.0 * h_true]
        for c, ct in zip(H.components, H_true):                  # pass(9)
            assert float((c - ct).norm()) < tol

        g2 = Chebfun2.from_function(lambda a, b: a + b,
                                    domain=(-2.0, 2.0, -2.0, 2.0))
        h = f.compose(g2)
        assert float((h - f).norm()) < tol                       # pass(10)
        H = f.compose([g2, g2, g2])
        for c in H.components:                                   # pass(11)
            assert float((c - f).norm()) < tol
