"""Port of MATLAB Chebfun tests/diskfunv/test_times_divide.m
(Fable 5).

FIXED: Diskfunv times/power added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/diskfunv/test_times_divide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv

THS = jnp.asarray(np.linspace(-np.pi, np.pi, 9))
RS = jnp.asarray(np.linspace(0.0, 1.0, 7))
TT, RR = jnp.meshgrid(THS, RS, indexing="ij")


def _v():
    return Diskfunv(
        Diskfun.from_function(
            lambda t, r: r * jnp.cos(t)),
        Diskfun.from_function(lambda t, r: r ** 2))


class TestDiskfunvTimesDivide:
    def test_times_scalar_field_componentwise(self):
        v = _v()
        # scalar
        w2 = v.times(2.0)
        f2, _ = w2(TT, RR)
        fv, _ = v(TT, RR)
        assert float(jnp.max(jnp.abs(f2 - 2 * fv))) < 1e-11
        # scalar Diskfun field
        s = Diskfun.from_function(lambda t, r: 1.0 + r * jnp.sin(t))
        ws = v.times(s)
        fs, _ = ws(TT, RR)
        assert float(jnp.max(jnp.abs(
            fs - fv * s(TT, RR)))) < 1e-9
        # componentwise square == power(2)
        wsq = v.times(v)
        fq, gq = wsq(TT, RR)
        fp, gp = v.power(2)(TT, RR)
        assert float(jnp.max(jnp.abs(fq - fp))) < 1e-9
        assert float(jnp.max(jnp.abs(gq - gp))) < 1e-9
