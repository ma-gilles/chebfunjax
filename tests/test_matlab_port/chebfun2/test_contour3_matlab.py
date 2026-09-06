"""Port of MATLAB Chebfun tests/chebfun2/test_contour3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_contour3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chebfunjax.chebfun2d.chebfun2 import chebfun2  # noqa: E402
from chebfunjax.plotting import contour3  # noqa: E402

jax.config.update("jax_enable_x64", True)


class TestChebfun2Contour3:
    def test_all_matlab_assertions(self):
        f = chebfun2(lambda x, y: jnp.cos(x * y))
        x = np.arange(-1, 1.0001, .1)
        xx, yy = np.meshgrid(x, x)
        contour3(f)
        contour3(f, 5)
        contour3(f, [0.8, 0.8])
        contour3(f, "numpts", 100)
        contour3(f, "pivots", "r.-")
        contour3(xx, yy, f)
        plt.close("all")
