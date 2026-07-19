"""Port of MATLAB Chebfun tests/deltafun/test_anyDelta.m (Opus 4.8).

MATLAB ``anyDelta(d)`` maps to chebfunjax ``Deltafun.has_deltas``.  chebfunjax
has no truly-empty Deltafun (a ``funPart`` is always required), so the empty
constructor is emulated by a Deltafun with no delta data.

Provenance
----------
MATLAB source : tests/deltafun/test_anyDelta.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DELTA_TOL = 1e-9  # pref.deltaPrefs.deltaTol
DOM = Domain((-1.0, 1.0))


def _f():
    return Bndfun.from_function(jnp.exp, DOM)


class TestDeltafunAnyDelta:
    def test_no_deltas(self):
        # pass(1): d = deltafun();  ~anyDelta(d)
        d = Deltafun.from_fun(_f())
        assert not d.has_deltas

    def test_random_block_has_deltas(self):
        # pass(2): mag = rand(5,5), loc = rand(1,5) -> anyDelta
        mag = np.random.rand(5, 5)
        loc = np.random.rand(5)
        d = Deltafun(_f(), jnp.asarray(loc), jnp.asarray(mag))
        assert d.has_deltas

    def test_single_unit_delta(self):
        # pass(3): mag = 1, loc = 0 -> anyDelta
        d = Deltafun(_f(), jnp.array([0.0]), jnp.array([1.0]))
        assert d.has_deltas

    def test_below_deltatol_is_not_a_delta(self):
        # pass(4): mag = deltaTol/2 -> ~anyDelta
        d = Deltafun(_f(), jnp.array([0.0]), jnp.array([DELTA_TOL / 2]))
        assert not d.has_deltas

    def test_empty_delta_data(self):
        # pass(5): mag = [], loc = [] -> ~anyDelta
        d = Deltafun.from_fun(_f())
        assert not d.has_deltas
