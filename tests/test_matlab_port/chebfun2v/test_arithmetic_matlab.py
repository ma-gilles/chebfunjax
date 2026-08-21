"""Port of MATLAB Chebfun tests/chebfun2v/test_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _maxdiff(F, fns, dom=(-1.0, 1.0, -1.0, 1.0)):
    xs = jnp.linspace(dom[0] + 1e-9, dom[1] - 1e-9, 9)
    ys = jnp.linspace(dom[2] + 1e-9, dom[3] - 1e-9, 9)
    X, Y = jnp.meshgrid(xs, ys)
    worst = 0.0
    for c, fn in zip(F.components, fns):
        f2 = Chebfun2(approx=c)
        worst = max(worst, float(jnp.max(jnp.abs(
            jnp.asarray(f2(X, Y)) - fn(X, Y)))))
    return worst


class TestChebfun2vArithmetic:
    def test_all_matlab_assertions(self):
        f = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.cos(x))
        g = Chebfun2v.from_functions(lambda x, y: jnp.sin(y),
                                     lambda x, y: jnp.sin(y))
        plus = lambda X, Y: jnp.cos(X) + jnp.sin(Y)
        minus = lambda X, Y: jnp.cos(X) - jnp.sin(Y)
        mult = lambda X, Y: jnp.cos(X) * jnp.sin(Y)
        assert _maxdiff(f + g, [plus, plus]) < TOL  # pass(1)
        assert _maxdiff(f - g, [minus, minus]) < TOL  # pass(2)
        assert _maxdiff(f * g, [mult, mult]) < TOL  # pass(3)
