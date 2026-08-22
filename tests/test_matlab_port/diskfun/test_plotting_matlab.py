"""Port of MATLAB Chebfun tests/diskfun/test_plotting.m (Fable 5).

MATLAB's ``ishold`` bookkeeping has no matplotlib counterpart (each
call draws on fresh axes unless one is passed), so pass(1) reduces to
the plots simply succeeding.  ``axis`` maps to ``get_xlim/get_ylim``.

Provenance
----------
MATLAB source : tests/diskfun/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from chebfunjax.diskfun.diskfun import Diskfun

jax.config.update("jax_enable_x64", True)


class TestDiskfunPlotting:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        f = Diskfun.from_function(
            lambda t, r: r ** 3 * jnp.sin(3 * t) + jnp.cos(4 * r ** 2))

        # pass(1): plot, surf, contour all succeed (and leave hold off).
        f.plot()
        f.surf()
        f.contour()
        plt.close("all")

        # pass(2): plot(f) has axis [-1 1 -1 1].
        fig, ax = f.plot()
        assert np.allclose(ax.get_xlim(), [-1.0, 1.0])
        assert np.allclose(ax.get_ylim(), [-1.0, 1.0])
        plt.close("all")

        # pass(3): contour(f) has axis [-1 1 -1 1].
        fig, ax = f.contour()
        assert np.allclose(ax.get_xlim(), [-1.0, 1.0])
        assert np.allclose(ax.get_ylim(), [-1.0, 1.0])
        plt.close("all")

        # pass(4): pivot plot with a linespec.
        f.plot(".-")
        plt.close("all")

        # pass(5): contour with explicit levels and linespec.
        f.contour(levels=[0.0, 0.0], fmt="k-")
        plt.close("all")
