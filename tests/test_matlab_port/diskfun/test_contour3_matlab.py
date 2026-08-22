"""Port of MATLAB Chebfun tests/diskfun/test_contour3.m (Fable 5).

Positional MATLAB options map to Python keywords: ``contour3(f, 5)`` ->
``levels=5``, ``('numpts', 100)`` -> ``n_pts=100``, ``('pivots',
'r.-')`` -> ``pivots='r.-'``, ``contour3(xx, yy, f)`` -> ``xx=, yy=``.

Provenance
----------
MATLAB source : tests/diskfun/test_contour3.m
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


class TestDiskfunContour3:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        f = Diskfun.from_function(
            lambda t, r: jnp.cos(jnp.cos(4 * r * jnp.cos(t)) ** 2
                                 + jnp.sin(5 * r * jnp.sin(t)) ** 2))
        x = np.arange(-np.pi, np.pi, 0.1)
        y = np.arange(-1.0, 1.0, 0.1)
        xx, yy = np.meshgrid(x, y)

        f.contour3()
        f.contour3(levels=5)
        f.contour3(levels=[0.4, 0.4])
        f.contour3(n_pts=100)
        f.contour3(pivots="r.-")
        f.contour3(xx=xx, yy=yy)
        plt.close("all")
