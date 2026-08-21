"""Port of MATLAB Chebfun tests/chebmatrix/test_waterfall.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_waterfall.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402

import chebfunjax as cj  # noqa: E402
from chebfunjax.operators.chebmatrix import ChebMatrix  # noqa: E402

jax.config.update("jax_enable_x64", True)


class TestChebmatrixWaterfall:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt
        Q = ChebMatrix([[
            cj.chebfun(lambda x: jnp.sin(jnp.pi * x)),
            cj.chebfun(lambda x: jnp.sin(jnp.pi * (x - 0.3))),
            cj.chebfun(lambda x: jnp.sin(jnp.pi * (x - 0.6))),
        ]])
        assert Q.waterfall() is not None  # pass(1)
        assert Q.waterfall([0, 0.3, 0.6]) is not None  # pass(2)
        assert Q.waterfall([0, 0.3, 0.6], LineWidth=2,
                           FaceAlpha=0.5,
                           FaceColor="r") is not None  # pass(3)
        plt.close("all")
