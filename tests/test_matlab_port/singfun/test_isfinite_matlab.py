"""Original native public predicate cases, Chebfun7574c77.

Source constructor data, endpoint hints, preferences and assertions are retained.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun.singfun import Singfun

A = 0.64
D = -1.28


def _sf(f, exps):
    hints = ("root", "none") if exps[0] > 0 else ("sing", "none")
    return Singfun.constructor(f, {"exponents": exps, "singType": hints}, ChebfunPref())


class TestSingfunIsfinite:
    def test_frac_root_left_is_finite(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        assert f.isfinite()

    def test_frac_pole_left_not_finite(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        assert not f.isfinite()
