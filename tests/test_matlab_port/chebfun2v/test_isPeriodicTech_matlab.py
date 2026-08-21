"""Port of MATLAB Chebfun tests/chebfun2v/test_isPeriodicTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_isPeriodicTech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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


class TestChebfun2vIsperiodictech:
    def test_all_matlab_assertions(self):
        f = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        assert not f.is_periodic_tech()  # pass(1)

        dom = (-1.0, 1.0, -np.pi, np.pi)
        f1 = Chebfun2.from_function(
            lambda x, y: jnp.cos(np.pi * x) + 0 * y, domain=dom,
            trig=True)
        f2 = Chebfun2.from_function(
            lambda x, y: jnp.sin(y) + 0 * x, domain=dom, trig=True)
        f = Chebfun2v([f1.approx, f2.approx])
        assert f.is_periodic_tech()  # pass(2)
        f = Chebfun2v([f1.approx, f1.approx, f2.approx])
        assert f.is_periodic_tech()  # pass(3)
