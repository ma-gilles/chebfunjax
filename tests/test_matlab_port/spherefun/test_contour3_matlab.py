"""Port of MATLAB Chebfun tests/spherefun/test_contour3.m (Fable 5).

Positional MATLAB options map to Python keywords: ``contour3(f, 5)`` ->
``levels=5``, ``('numpts', 100)`` -> ``n_pts=100``.

Provenance
----------
MATLAB source : tests/spherefun/test_contour3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

import matplotlib

matplotlib.use("Agg")

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)


class TestSpherefunContour3:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        f = Spherefun.sphharm(4, 3)

        f.contour3()
        f.contour3(levels=5)
        f.contour3(levels=[0.3, 0.3])
        f.contour3(n_pts=100)
        plt.close("all")
