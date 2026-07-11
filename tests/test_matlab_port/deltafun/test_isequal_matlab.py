"""Port of MATLAB Chebfun tests/deltafun/test_isequal.m (Fable 5).

FIXED: Deltafun.isequal added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/deltafun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

D = Domain((-1.0, 1.0))


def _mk(mags, locs):
    f = Bndfun.from_function(jnp.sin, D)
    return Deltafun.from_fun_and_deltas(f, jnp.asarray(locs),
                                        jnp.asarray(mags))


class TestDeltafunIsequal:
    def test_self_equality(self):
        f = _mk([[1.0, -2.0]], [0.25, 0.5])
        assert f.isequal(f)

    def test_different_magnitudes(self):
        f = _mk([[1.0]], [0.25])
        g = _mk([[1.5]], [0.25])
        assert not f.isequal(g)

    def test_non_deltafun(self):
        f = _mk([[1.0]], [0.25])
        assert not f.isequal("not a deltafun")
