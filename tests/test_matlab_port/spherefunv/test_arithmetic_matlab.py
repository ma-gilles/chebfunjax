"""Port of MATLAB Chebfun tests/spherefunv/test_arithmetic.m
(Fable 5).

FIXED: Spherefunv times/power added in the Fable 5 audit
(plus/minus/uminus already existed).  chebfunjax Spherefunv is the
2-component tangential representation; the arithmetic identities
asserted here are representation-independent.

Provenance
----------
MATLAB source : tests/spherefunv/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

LAMS = jnp.asarray(np.linspace(-3.0, 3.0, 9))
THS = jnp.asarray(np.linspace(0.1, 3.0, 9))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")


def _v():
    return Spherefunv(
        Spherefun.from_function(
            lambda lam, th: jnp.cos(th) * jnp.sin(lam)),
        Spherefun.from_function(lambda lam, th: jnp.cos(th)))


def _maxdiff(a, b):
    fa, ga = a(LL, TT)
    fb, gb = b(LL, TT)
    return max(float(jnp.max(jnp.abs(fa - fb))),
               float(jnp.max(jnp.abs(ga - gb))))


class TestSpherefunvArithmetic:
    def test_plus_minus_uminus_times(self):
        v = _v()
        assert _maxdiff(v + v, v.times(2.0)) < 1e-11
        w = v - v
        fw, gw = w(LL, TT)
        assert float(jnp.max(jnp.abs(fw))) < 1e-12
        assert float(jnp.max(jnp.abs(gw))) < 1e-12
        assert _maxdiff(-v, v.times(-1.0)) < 1e-12
        assert _maxdiff(v.times(v), v.power(2)) < 1e-9
