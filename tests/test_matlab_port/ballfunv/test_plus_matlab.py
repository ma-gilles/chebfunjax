"""Port of MATLAB Chebfun tests/ballfunv/test_plus.m (Fable 5).

FIXED: Ballfunv arithmetic exists / was added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/ballfunv/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

RS = jnp.asarray(np.linspace(0.05, 1.0, 5))
LS = jnp.asarray(np.linspace(-3, 3, 5))
TS = jnp.asarray(np.linspace(0.1, 3.0, 5))
RR, LL, TT = jnp.meshgrid(RS, LS, TS, indexing="ij")


def _v():
    return Ballfunv(
        Ballfun.from_function(lambda x, y, z: x),
        Ballfun.from_function(lambda x, y, z: y * z),
        Ballfun.from_function(lambda x, y, z: z + x * y))


def _maxdiff(a, b):
    return max(
        float(jnp.max(jnp.abs(ca(RR, LL, TT) - cb(RR, LL, TT))))
        for ca, cb in zip(a.components, b.components))


class TestBallfunvPlus:
    def test_plus(self):
        v = _v()
        w = v + v
        assert _maxdiff(w, v.times(2.0)) < 1e-12
