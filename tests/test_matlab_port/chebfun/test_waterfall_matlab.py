"""Port of MATLAB Chebfun tests/chebfun/test_waterfall.m (Fable 5).

MATLAB's ``waterfall(Q)`` on an array-valued chebfun / quasimatrix maps
to :func:`chebfunjax.plotting.waterfall` on the list of columns.  As in
MATLAB, the assertions only check that nothing crashes.

Provenance
----------
MATLAB source : tests/chebfun/test_waterfall.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")

import chebfunjax as cj
import chebfunjax.plotting as P

jax.config.update("jax_enable_x64", True)


def _does_not_crash(fn):
    import matplotlib.pyplot as plt
    try:
        fn()
        return True
    finally:
        plt.close("all")


class TestChebfunWaterfall:
    def test_all_matlab_assertions(self):
        Q = [cj.chebfun(lambda x, s=s: jnp.sin(jnp.pi * (x - s)))
             for s in (0.0, 0.3, 0.6)]
        assert _does_not_crash(lambda: P.waterfall(Q))                  # 1
        assert _does_not_crash(
            lambda: P.waterfall([cj.chebfun(lambda x: 0.0 * x)]))       # 2
        assert _does_not_crash(lambda: P.waterfall(Q))                  # 3
        assert _does_not_crash(lambda: P.waterfall(Q, [0.0, 0.3, 0.6]))  # 4
        assert _does_not_crash(lambda: P.waterfall(Q))                  # 5
        assert _does_not_crash(
            lambda: P.waterfall(Q, linewidth=2, FaceAlpha=0.5,
                                FaceColor="r"))                         # 6
