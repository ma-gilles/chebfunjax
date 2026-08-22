"""Port of MATLAB Chebfun tests/chebfun/test_plotcoeffs.m (Fable 5).

MATLAB's ``plotcoeffs`` maps to :func:`chebfunjax.plotting.plotcoeffs`;
the positional 'loglog' flag and linespec map to the ``loglog``/``fmt``
keywords, and array-valued chebfuns / quasimatrices map to lists of
columns.  As in MATLAB, the assertions only check that nothing crashes.

Provenance
----------
MATLAB source : tests/chebfun/test_plotcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import chebfunjax as cj
import chebfunjax.plotting as P
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain

jax.config.update("jax_enable_x64", True)


def _does_not_crash(fn):
    import matplotlib.pyplot as plt
    try:
        fn()
        return True
    finally:
        plt.close("all")


class TestChebfunPlotcoeffs:
    def test_all_matlab_assertions(self):
        f = cj.chebfun(jnp.sin)
        p1 = cj.chebfun(jnp.sin, domain=(-1.0, 0.0))
        p2 = cj.chebfun(jnp.exp, domain=(0.0, 1.0))
        g = Chebfun(funs=list(p1.funs) + list(p2.funs),
                    domain=Domain((-1.0, 0.0, 1.0)))
        F = [cj.chebfun(jnp.sin), cj.chebfun(jnp.cos), cj.chebfun(jnp.exp)]
        G = [Chebfun(funs=list(cj.chebfun(fn, domain=(-1.0, 0.0)).funs)
                     + list(cj.chebfun(fn, domain=(0.0, 1.0)).funs),
                     domain=Domain((-1.0, 0.0, 1.0)))
             for fn in (jnp.sin, jnp.cos, jnp.exp)]
        Q = F

        assert _does_not_crash(lambda: P.plotcoeffs(f))            # 1
        assert _does_not_crash(lambda: P.plotcoeffs(g))            # 2
        assert _does_not_crash(lambda: P.plotcoeffs(F))            # 3
        assert _does_not_crash(lambda: P.plotcoeffs(G))            # 4
        assert _does_not_crash(lambda: P.plotcoeffs(Q))            # 5

        # Plot flags and options.
        assert _does_not_crash(lambda: P.plotcoeffs(g, loglog=True))  # 6
        assert _does_not_crash(lambda: P.plotcoeffs(g, fmt=".--"))    # 7

        # Trigtech chebfuns.
        ftrig = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), trig=True)
        Ftrig = [cj.chebfun(lambda x: jnp.sin(jnp.pi * x), trig=True),
                 cj.chebfun(lambda x: jnp.cos(jnp.pi * x), trig=True)]
        assert _does_not_crash(lambda: P.plotcoeffs(ftrig))            # 8
        assert _does_not_crash(lambda: P.plotcoeffs(Ftrig))            # 9
        assert _does_not_crash(lambda: P.plotcoeffs(ftrig, loglog=True))  # 10
        assert _does_not_crash(lambda: P.plotcoeffs(ftrig, fmt=".--"))    # 11
