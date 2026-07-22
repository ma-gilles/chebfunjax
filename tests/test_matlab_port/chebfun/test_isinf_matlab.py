"""Port of MATLAB Chebfun tests/chebfun/test_isinf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsinf:
    def test_smooth_not_inf(self):
        f = cj.chebfun(lambda x: jnp.exp(x))
        assert not bool(f.isinf())

    def test_unbounded_not_inf(self):
        # pass(5): ~isinf of a smooth, bounded function on [0, inf).
        f = cj.chebfun(lambda x: 0.75 + jnp.sin(10 * x) / jnp.exp(x),
                       domain=(0, jnp.inf))
        assert not bool(f.isinf())

    def test_singular_is_inf(self):
        # pass(4): a singular SingFun (endpoint pole) is infinite.
        dom = (-2.0, 7.0)
        pow = -1.64
        f = cj.chebfun(lambda x: jnp.sin(100 * x) * (x - dom[0]) ** pow,
                       domain=dom, exps=(pow, 0.0))
        assert bool(f.isinf())

    def test_blowup_cases(self):
        # pass(3): pointValues(1,1) = Inf (no pointValues field in chebfunjax);
        # pass(6): 'exps' [0 -1] blow-up on the unbounded domain (-inf, -3*pi].
        # Neither is representable (no pointValues field / no unbounded SingFun).
        pytest.skip("chebfun-level pointValues=Inf / unbounded SingFun blow-up "
                    "not implemented")
