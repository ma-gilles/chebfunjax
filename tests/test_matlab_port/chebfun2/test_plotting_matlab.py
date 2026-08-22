"""Port of MATLAB Chebfun tests/chebfun2/test_plotting.m (Fable 5).

MATLAB's ``ishold`` bookkeeping has no matplotlib counterpart, so
pass(1) reduces to the plots simply succeeding.

Provenance
----------
MATLAB source : tests/chebfun2/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

jax.config.update("jax_enable_x64", True)


class TestChebfun2Plotting:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        # pass(1): plot, surf, contour, waterfall, surf all succeed.
        f = Chebfun2.from_function(lambda x, y: jnp.exp(jnp.cos(10 * x * y)))
        f.plot()
        f.surf()
        f.contour()
        f.waterfall()
        f.surf()
        plt.close("all")

        # pass(2): the same on a non-default domain.
        f = Chebfun2.from_function(lambda x, y: x * y,
                                   domain=(-1.0, 2.0, -1.0, 2.0))
        f.plot()
        f.surf()
        f.contour()
        f.waterfall()
        plt.close("all")

        # pass(3): waterfall with a linespec.
        h = lambda x, y: (
            0.75 * jnp.exp(-((9 * x - 2) ** 2 + (9 * y - 2) ** 2) / 4)
            + 0.75 * jnp.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
            + 0.5 * jnp.exp(-((9 * x - 7) ** 2 + (9 * y - 3) ** 2) / 4)
            - 0.2 * jnp.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2))
        H = Chebfun2.from_function(h)
        H.waterfall("-")
        plt.close("all")
