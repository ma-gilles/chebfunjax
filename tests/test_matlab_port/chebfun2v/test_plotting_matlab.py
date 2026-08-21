"""Port of MATLAB Chebfun tests/chebfun2v/test_plotting.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_plotting.m
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

import matplotlib

matplotlib.use("Agg")


class TestChebfun2vPlotting:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        G = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y,
                                     lambda x, y: y)
        assert F.quiver() is not None
        assert G.quiver3() is not None
        assert G.surf() is not None
        plt.close("all")
