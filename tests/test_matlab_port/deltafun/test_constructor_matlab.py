"""Port of MATLAB Chebfun tests/deltafun/test_constructor.m (Opus 4.8).

MATLAB ``deltafun(f, struct('deltaMag', ..., 'deltaLoc', ...))`` maps to
chebfunjax ``Deltafun(funPart, delta_locs, delta_mags)``.  Note MATLAB stores
``deltaLoc`` as a 1xN row; chebfunjax ravels it to shape (N,).

Provenance
----------
MATLAB source : tests/deltafun/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DELTA_TOL = 1e-9  # pref.deltaPrefs.deltaTol
DOM = Domain((-1.0, 1.0))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _zero_bndfun(dom=DOM):
    return Bndfun.from_function(lambda x: jnp.zeros_like(x), dom)


class TestDeltafunConstructor:
    @pytest.mark.skip(
        reason="chebfunjax has no empty Deltafun: a funPart is always required, "
        "so deltafun() and its isempty check have no analog"
    )
    def test_empty_constructor_isempty(self):
        # pass(1): d = deltafun(); isempty(d) && isa(d,'deltafun')
        pass

    def test_zero_funpart_no_deltas(self):
        # pass(2): deltafun(fun.constructor(0)) -> empty deltas, iszero(funPart)
        d = Deltafun.from_fun(_zero_bndfun())
        assert d.n_deltas == 0
        assert not d.has_deltas
        assert float(jnp.max(jnp.abs(d.funPart.coeffs))) == 0.0  # iszero funPart

    def test_nonzero_funpart_no_deltas(self):
        # pass(3): deltafun(sin on [0,1]) -> empty deltas, ~iszero(funPart)
        f = Bndfun.from_function(jnp.sin, Domain((0.0, 1.0)))
        d = Deltafun.from_fun(f)
        assert d.n_deltas == 0
        assert float(jnp.max(jnp.abs(d.funPart.coeffs))) > 0.0

    def test_shapes_3x3(self):
        # pass(4): deltaMag rand(3,3), deltaLoc rand(1,3) -> [3,3], [1,3]
        d = Deltafun(_zero_bndfun(), jnp.asarray(np.random.rand(3)),
                     jnp.asarray(np.random.rand(3, 3)))
        assert d.delta_mags.shape == (3, 3)
        assert d.delta_locs.shape == (3,)

    def test_shapes_5x5_column_loc(self):
        # pass(5): deltaMag rand(5,5), deltaLoc rand(5,1) -> [5,5], [1,5]
        d = Deltafun(_zero_bndfun(), jnp.asarray(np.random.rand(5, 1)),
                     jnp.asarray(np.random.rand(5, 5)))
        assert d.delta_mags.shape == (5, 5)
        assert d.delta_locs.shape == (5,)

    def test_data_stored_exactly(self):
        # pass(6): funPart == f, deltaMag exact, deltaLoc exact
        a, b = 1.3, -0.7  # stand in for randn scalars
        f = Bndfun.from_function(lambda x: 20 * a * jnp.sin(10 * b * x), DOM)
        deltas = np.random.rand(5, 5)
        locs = np.linspace(-0.9, 0.9, 5)
        d = Deltafun(f, jnp.asarray(locs), jnp.asarray(deltas))
        X = jnp.asarray(np.linspace(-1.0, 1.0, 40))
        assert _ninf(f(X) - d.funPart(X)) == 0.0
        assert _ninf(d.delta_mags - deltas) == 0.0
        assert _ninf(d.delta_locs - locs) == 0.0

    def test_data_stored_exactly_with_pref(self):
        # pass(7): identical to pass(6) but with a pref argument (ignored here)
        a, b = 1.3, -0.7
        f = Bndfun.from_function(lambda x: 20 * a * jnp.sin(10 * b * x), DOM)
        deltas = np.random.rand(5, 5)
        locs = np.linspace(-0.9, 0.9, 5)
        d = Deltafun(f, jnp.asarray(locs), jnp.asarray(deltas))
        X = jnp.asarray(np.linspace(-1.0, 1.0, 40))
        assert _ninf(f(X) - d.funPart(X)) == 0.0
        assert _ninf(d.delta_mags - deltas) == 0.0
        assert _ninf(d.delta_locs - locs) == 0.0

    def test_deltatol_cleaning(self):
        # pass(8): tiny magnitudes removed at construction
        f = Bndfun.from_function(lambda x: 20 * 1.3 * jnp.sin(10 * -0.7 * x), DOM)
        mag = np.array([-1.0, DELTA_TOL / 2, 1.0, DELTA_TOL / 2])
        loc = np.array([-1.0, 0.0, 0.5, 1.0])
        d = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        assert _ninf(d.delta_mags - np.array([-1.0, 1.0])) == 0.0
        assert _ninf(d.delta_locs - np.array([-1.0, 0.5])) == 0.0
