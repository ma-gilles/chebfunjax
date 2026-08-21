"""Port of MATLAB Chebfun tests/chebmatrix/test_plotcoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_plotcoeffs.m
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


class TestChebmatrixPlotcoeffs:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt
        f = cj.chebfun(lambda x: jnp.sin(jnp.pi * x))
        g = cj.chebfun(lambda x: jnp.cos(jnp.pi * x))
        Q = ChebMatrix([[f], [g], [1.5]])
        assert Q.plotcoeffs() is not None
        plt.close("all")
