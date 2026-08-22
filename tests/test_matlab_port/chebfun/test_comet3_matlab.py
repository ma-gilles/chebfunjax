"""Port of MATLAB Chebfun tests/chebfun/test_comet3.m (Fable 5).

MATLAB's ``comet3`` maps to :func:`chebfunjax.plotting.comet3`;
array-valued chebfuns / quasimatrices map to lists of columns.  As in
MATLAB, the assertions only check that nothing crashes.

Provenance
----------
MATLAB source : tests/chebfun/test_comet3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
import chebfunjax.plotting as P

jax.config.update("jax_enable_x64", True)


def _pw(fn, brks):
    funs = []
    for a, b in zip(brks[:-1], brks[1:]):
        funs.extend(cj.chebfun(fn, domain=(a, b)).funs)
    return Chebfun(funs=funs, domain=Domain(tuple(brks)))


def _does_not_crash(fn):
    import matplotlib.pyplot as plt
    try:
        fn()
        return True
    finally:
        plt.close("all")


class TestChebfunComet3:
    def test_all_matlab_assertions(self):
        fsr1 = _pw(jnp.sin, [-1.0, 0.0, 1.0])
        fsr2 = _pw(jnp.cos, [-1.0, 0.5, 1.0])
        fsr3 = _pw(jnp.exp, [-1.0, -0.5, 1.0])
        far1 = [_pw(jnp.sin, [-1.0, 0.0, 1.0]),
                _pw(jnp.cos, [-1.0, 0.0, 1.0]),
                _pw(jnp.exp, [-1.0, 0.0, 1.0])]
        fqr1 = far1

        assert _does_not_crash(lambda: P.comet3(fsr1, fsr2, fsr3))  # 1
        assert _does_not_crash(lambda: P.comet3(far1))              # 2
        assert _does_not_crash(lambda: P.comet3(fqr1))              # 3
