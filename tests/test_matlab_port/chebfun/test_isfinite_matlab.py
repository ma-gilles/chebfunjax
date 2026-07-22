"""Port of MATLAB Chebfun tests/chebfun/test_isfinite.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsfinite:
    def test_smooth_is_finite(self):
        # pass(1): a smooth function is finite.
        f = cj.chebfun(jnp.sin)
        assert bool(f.isfinite())
        assert not bool(f.isinf())
        assert not bool(f.isnan())

    def test_singular_not_finite(self):
        # pass(4): a singular SingFun (endpoint pole) is not finite.
        dom = (-2.0, 7.0)
        pow = -1.64
        f = cj.chebfun(lambda x: jnp.sin(100 * x) * (x - dom[0]) ** pow,
                       domain=dom, exps=(pow, 0.0))
        assert not bool(f.isfinite())

    def test_blowup_cases(self):
        # pass(3): pointValues(1,1) = Inf (no pointValues field in chebfunjax);
        # pass(6): 'exps' [0 -1] blow-up on the unbounded domain (-inf, -3*pi].
        pytest.skip("chebfun-level pointValues=Inf / unbounded SingFun blow-up "
                    "not implemented")
