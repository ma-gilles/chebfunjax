"""Port of MATLAB Chebfun tests/spherefun/test_plotting.m (Fable 5).

MATLAB's ``ishold`` bookkeeping has no matplotlib counterpart, so
pass(1) reduces to the plots simply succeeding.  ``axis`` maps to
``get_xlim/get_ylim/get_zlim``.

Provenance
----------
MATLAB source : tests/spherefun/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)


class TestSpherefunPlotting:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        f = Spherefun.from_function(
            lambda lam, th: 1.0 - jnp.exp(
                jnp.sin(lam) * jnp.cos(lam) * jnp.sin(th) ** 2))

        # pass(1): plot, surf, contour all succeed.
        f.plot()
        f.surf()
        f.contour()
        plt.close("all")

        # pass(2): plot(f) has axis [-1 1 -1 1 -1 1].
        fig, ax = f.plot()
        assert np.allclose(ax.get_xlim(), [-1.0, 1.0])
        assert np.allclose(ax.get_ylim(), [-1.0, 1.0])
        assert np.allclose(ax.get_zlim(), [-1.0, 1.0])
        plt.close("all")

        # pass(3): contour(f) has axis [-1 1 -1 1 -1 1].
        fig, ax = f.contour()
        assert np.allclose(ax.get_xlim(), [-1.0, 1.0])
        assert np.allclose(ax.get_ylim(), [-1.0, 1.0])
        assert np.allclose(ax.get_zlim(), [-1.0, 1.0])
        plt.close("all")

        # pass(4): pivot plot with a linespec.
        f.plot(".-")
        plt.close("all")

        # pass(5): contour with explicit levels and linespec.
        f.contour(levels=[0.0, 0.0], fmt="k-")
        plt.close("all")
