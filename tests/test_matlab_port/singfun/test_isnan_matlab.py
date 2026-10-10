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


class TestSingfunIsnan:
    def test_frac_root_left_not_nan(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        assert not f.isnan()

    def test_frac_pole_left_not_nan(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        assert not f.isnan()

    def test_nan_times_singfun_is_nan(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        g = float("nan") * f
        assert g.isnan()
