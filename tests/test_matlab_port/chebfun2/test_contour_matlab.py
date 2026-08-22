"""Port of MATLAB Chebfun tests/chebfun2/test_contour.m (Fable 5).

MATLAB positional options map to Python keywords: ``contour(f, 5)`` ->
``levels=5``, ``('numpts', 100)`` -> ``n_pts=100``, ``('pivots',
'r.-')`` -> ``pivots='r.-'``, ``contour(xx, yy, f)`` -> ``xx=, yy=``;
``contourf`` -> ``filled=True``.

Provenance
----------
MATLAB source : tests/chebfun2/test_contour.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import chebfunjax.plotting as P
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

jax.config.update("jax_enable_x64", True)


class TestChebfun2Contour:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        x = np.arange(-1.0, 1.05, 0.1)
        xx, yy = np.meshgrid(x, x)

        P.contour(f)
        P.contour(f, levels=5)
        P.contour(f, levels=[0.0, 0.0])
        P.contour(f, n_pts=100)
        P.contour(f, pivots="r.-")
        P.contour(f, xx=xx, yy=yy)
        P.contour(f, filled=True)
        P.contour(f, levels=5, filled=True)
        P.contour(f, levels=[0.0, 0.0], filled=True)
        P.contour(f, n_pts=100, filled=True)
        P.contour(f, pivots=".", filled=True)
        P.contour(f, xx=xx, yy=yy, filled=True)
        plt.close("all")
