"""Port of MATLAB Chebfun tests/chebfun2/test_surf.m (Fable 5).

MATLAB's positional options map to Python keywords: ``surf(f, 'numpts',
10)`` -> ``n_pts=10``; ``surf(x, y, f)`` passes the coordinate
chebfun2s positionally.

Provenance
----------
MATLAB source : tests/chebfun2/test_surf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import chebfunjax.plotting as P

jax.config.update("jax_enable_x64", True)


class TestChebfun2Surf:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt

        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        x = Chebfun2.from_function(lambda x, y: x)
        y = Chebfun2.from_function(lambda x, y: y)

        P.surf(f)
        P.surf(f, f)
        P.surf(f, f, f)
        P.surf(f, n_pts=10)
        P.surf(x, y, f)
        P.surf(x, y, f, edgecolor="r")
        plt.close("all")
