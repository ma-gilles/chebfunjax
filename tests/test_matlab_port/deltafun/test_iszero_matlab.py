"""Port of MATLAB Chebfun tests/deltafun/test_iszero.m (Fable 5).

FIXED: Deltafun.iszero added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/deltafun/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

D = Domain((-1.0, 1.0))


class TestDeltafunIszero:
    def test_zero_distribution(self):
        assert Deltafun.zero_delta_fun().iszero()

    def test_nonzero_smooth_part(self):
        f = Deltafun.from_fun(Bndfun.from_function(jnp.sin, D))
        assert not f.iszero()

    def test_nonzero_delta(self):
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        f = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.5]),
                                         jnp.asarray([[1.0]]))
        assert not f.iszero()

    def test_below_deltaTol_is_zero(self):
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        f = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.5]),
                                         jnp.asarray([[1e-12]]))
        assert f.iszero()
