"""Port of MATLAB Chebfun tests/chebfun/test_assignColumns.m (Fable 5).

``assign_columns(cols, g)`` replaces the given (0-based) columns of an
array-valued chebfun with the columns of ``g``; ``g=None`` deletes columns.
chebfunjax requires ``g`` to share ``f``'s breakpoints, so the ported cases
build ``g`` on ``f``'s domain (the resulting functions are identical to
MATLAB's, which builds ``g`` on a coarser domain and unifies breakpoints).

The ``':'`` selector, the constant-value assignment, the row-chebfun
(transpose) case, the grow-beyond-dimension case, the error-identifier
cases, the quasimatrix loop, and the unbounded-domain case have no
chebfunjax counterpart and stay skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=10) - 1)
_DOM = (-1, 0, 1)


def _f():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1), domain=_DOM
    )


class TestChebfunAssigncolumns:
    def test_assign_first_column(self):
        # pass(k,1): assignColumns(f, 1, g) == [x cos exp].
        # FIXED (Fable 5, Big-Three array-valued epic).
        h = _f().assign_columns(0, cj.chebfun(lambda x: x, domain=_DOM))
        exact = jnp.stack([X, jnp.cos(X), jnp.exp(X)], axis=-1)
        assert float(jnp.max(jnp.abs(h(X) - exact))) < 10 * h.vscale * EPS

    def test_assign_last_column(self):
        # pass(k,2): assignColumns(f, 3, g) == [sin cos x].
        # FIXED (Fable 5, Big-Three array-valued epic).
        h = _f().assign_columns(2, cj.chebfun(lambda x: x, domain=_DOM))
        exact = jnp.stack([jnp.sin(X), jnp.cos(X), X], axis=-1)
        assert float(jnp.max(jnp.abs(h(X) - exact))) < 10 * h.vscale * EPS

    def test_assign_reorder(self):
        # pass(k,3): assignColumns(f, [3 1], [x x^2]) == [x^2 cos x].
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = cj.chebfun(lambda x: jnp.stack([x, x**2], axis=-1), domain=_DOM)
        h = _f().assign_columns([2, 0], g)
        exact = jnp.stack([X**2, jnp.cos(X), X], axis=-1)
        assert float(jnp.max(jnp.abs(h(X) - exact))) < 10 * h.vscale * EPS

    def test_assign_repeated_target(self):
        # pass(k,4): assignColumns(f, [2 2], [x x^2]) == [sin x^2 exp] (last wins).
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = cj.chebfun(lambda x: jnp.stack([x, x**2], axis=-1), domain=_DOM)
        h = _f().assign_columns([1, 1], g)
        exact = jnp.stack([jnp.sin(X), X**2, jnp.exp(X)], axis=-1)
        assert float(jnp.max(jnp.abs(h(X) - exact))) < 10 * h.vscale * EPS

    def test_assign_all_columns(self):
        # pass(k,5): assignColumns(f, [1 2 3], [x x^2 x^3]) == [x x^2 x^3].
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = cj.chebfun(lambda x: jnp.stack([x, x**2, x**3], axis=-1), domain=_DOM)
        h = _f().assign_columns([0, 1, 2], g)
        exact = jnp.stack([X, X**2, X**3], axis=-1)
        assert float(jnp.max(jnp.abs(h(X) - exact))) < 10 * h.vscale * EPS

    def test_delete_column(self):
        # pass(k,14,15): xx(:,2) = [] removes a column; numColumns == 1.
        # FIXED (Fable 5, Big-Three array-valued epic): assign_columns(col, None).
        xx = cj.chebfun(lambda t: jnp.stack([t, 2 * t], axis=-1))
        reduced = xx.assign_columns(1, None)
        assert reduced.n_columns == 1
        assert float(jnp.max(jnp.abs(jnp.ravel(reduced(X)) - X))) < EPS

    def test_colon_selector(self):
        # pass(k,6): assignColumns(f, ':', g).
        pytest.skip("Chebfun.assign_columns() does not accept the ':' selector (pass an "
                    "explicit column list)")

    def test_constant_assignment(self):
        # pass(k,7,8): assignColumns(f, [2 1], [-0.5 0.5]) assigns constant values.
        pytest.skip("Chebfun.assign_columns() requires a Chebfun operand, not a numeric "
                    "constant array")

    def test_error_conditions(self):
        # pass(k,9,10,11): numCols / domain mismatch raise.
        pytest.skip("assign_columns raises on breakpoint mismatch but not with MATLAB's "
                    "numCols/domain error identifiers")

    def test_grow_beyond_dimension(self):
        # pass(k,12): assignColumns(f, [1 2 4], g) grows to 4 columns.
        pytest.skip("Chebfun.assign_columns() does not grow the column count beyond "
                    "n_columns")

    def test_unbounded(self):
        # pass(k,13): assignColumns on an unbounded domain.
        pytest.skip("chebfunjax has no unbounded-domain support")
