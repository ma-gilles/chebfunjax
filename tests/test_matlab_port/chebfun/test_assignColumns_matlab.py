"""Mapped predicates from MATLAB Chebfun test_assignColumns.

``assign_columns(cols, g)`` replaces the given (0-based) columns of an
array-valued chebfun with the columns of ``g``; ``g=None`` deletes columns.
Finite clauses 1--12 use native query samples and breakpoint layouts for the
Chebfun-valued outer loop. Clause13 uses captured native continuation probes
and the original matrix infinity norm. The Quasimatrix loop is separate.

Provenance
----------
MATLAB source : tests/chebfun/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
_FIXTURE = Path(__file__).parent / "fixtures" / "abs_seed6178_native.json"
X = jnp.asarray(json.loads(_FIXTURE.read_text())["x"][:10])
_DOM = (-1, 0, 1)


def _source_error_norm(err):
    """Source norm(err(:),inf) for finite clauses 1--12."""
    return jnp.max(jnp.abs(jnp.ravel(err)))


def _f():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1), domain=_DOM
    )


class TestChebfunAssigncolumns:
    def test_assign_first_column(self):
        # pass(k,1): assignColumns(f, 1, g) == [x cos exp].
        # FIXED (Fable 5, Big-Three array-valued epic).
        # Native g uses the coarser domain [-1, 1], then overlap unifies it.
        h = _f().assign_columns(0, cj.chebfun(lambda x: x, domain=(-1, 1)))
        exact = jnp.stack([X, jnp.cos(X), jnp.exp(X)], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_assign_last_column(self):
        # pass(k,2): assignColumns(f, 3, g) == [sin cos x].
        # FIXED (Fable 5, Big-Three array-valued epic).
        h = _f().assign_columns(2, cj.chebfun(lambda x: x, domain=(-1, 1)))
        exact = jnp.stack([jnp.sin(X), jnp.cos(X), X], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_assign_reorder(self):
        # pass(k,3): assignColumns(f, [3 1], [x x^2]) == [x^2 cos x].
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = cj.chebfun(lambda x: jnp.stack([x, x**2], axis=-1), domain=_DOM)
        h = _f().assign_columns([2, 0], g)
        exact = jnp.stack([X**2, jnp.cos(X), X], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_assign_repeated_target(self):
        # pass(k,4): assignColumns(f, [2 2], [x x^2]) == [sin x^2 exp] (last wins).
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = cj.chebfun(lambda x: jnp.stack([x, x**2], axis=-1), domain=_DOM)
        h = _f().assign_columns([1, 1], g)
        exact = jnp.stack([jnp.sin(X), X**2, jnp.exp(X)], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_assign_all_columns(self):
        # pass(k,5): assignColumns(f, [1 2 3], [x x^2 x^3]) == [x x^2 x^3].
        # FIXED (Fable 5, Big-Three array-valued epic).
        g = cj.chebfun(lambda x: jnp.stack([x, x**2, x**3], axis=-1),
                       domain=(-1, -0.5, 0.5, 1))
        h = _f().assign_columns([0, 1, 2], g)
        exact = jnp.stack([X, X**2, X**3], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_delete_column(self):
        # Native pass(1,14); piecewise pass15 is in the focused source module.
        x = cj.chebfun(lambda t: t)
        xx = cj.Chebfun.horzcat(x, 2 * x)
        reduced = xx.assign_columns(1, None)
        assert reduced.n_columns == 1
        assert float((x - reduced).norm()) < EPS

    def test_colon_selector(self):
        # pass(k,6): assignColumns(f, ':', g) matches the explicit list.
        # g is built on its own breakpoints, which are unified by overlap.
        g = cj.chebfun(lambda x: jnp.stack([x, x**2, x**3], axis=-1),
                       domain=(-1.0, -0.5, 0.5, 1.0))
        h = _f().assign_columns(":", g)
        hc = _f().assign_columns([0, 1, 2], g)
        exact = jnp.stack([X, X**2, X**3], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS
        assert h.isequal(hc)

    def test_constant_assignment(self):
        # pass(k,7): assignColumns(f, [2 1], [-0.5 0.5]) assigns constants,
        # so column 2 becomes -0.5 and column 1 becomes 0.5.
        h = _f().assign_columns([1, 0], [-0.5, 0.5])
        exact = jnp.stack([jnp.full_like(X, 0.5), jnp.full_like(X, -0.5),
                           jnp.exp(X)], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_constant_assignment_row_chebfun(self):
        # pass(k,8): the same on the row (transposed) chebfun.
        h = _f().T.assign_columns([1, 0], jnp.array([[-0.5], [0.5]]))
        assert h.is_transposed
        exact = jnp.stack([jnp.full_like(X, 0.5), jnp.full_like(X, -0.5),
                           jnp.exp(X)], axis=-1).T
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_error_conditions(self):
        # pass(k,9,10,11): compare native error identifiers.
        g = cj.chebfun(lambda x: jnp.stack([x, x**2, x**3], axis=-1),
                       domain=(-1.0, -0.5, 0.5, 1.0))
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:assignColumns:numCols"):
            _f().assign_columns([0, 1, 2], g.T)
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:assignColumns:numCols"):
            _f().assign_columns([0, 1], g)
        g2 = cj.chebfun(lambda x: jnp.stack([x, x**2, x**3], axis=-1),
                        domain=(0.0, 1.0))
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:assignColumns:domain"):
            _f().assign_columns([0, 1, 2], g2)

    def test_grow_beyond_dimension(self):
        # pass(k,12): assignColumns(f, [1 2 4], g) grows f to 4 columns,
        # leaving the untouched third column as exp.
        g = cj.chebfun(lambda x: jnp.stack([x, x**2, x**3], axis=-1),
                       domain=(-1.0, -0.5, 0.5, 1.0))
        h = _f().assign_columns([0, 1, 3], g)
        assert h.n_columns == 4
        exact = jnp.stack([X, X**2, jnp.exp(X), X**3], axis=-1)
        assert float(_source_error_norm(h(X) - exact)) < 10 * h.vscale * EPS

    def test_unbounded(self):
        # pass(k,13): assignColumns(f, 2, g) on (-inf, -3*pi].
        # op = [exp(x) x*exp(x) (1-exp(x))/x]; g = exp(-x^2) replaces col 2;
        # oph = [exp(x) exp(-x^2) (1-exp(x))/x].
        dom = (-jnp.inf, -3 * np.pi)
        op = lambda x: jnp.stack(
            [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1)
        oph = lambda x: jnp.stack(
            [jnp.exp(x), jnp.exp(-x**2), (1 - jnp.exp(x)) / x], axis=-1)
        f = cj.chebfun(op, domain=dom)
        g = cj.chebfun(lambda x: jnp.exp(-x**2), domain=dom)
        h = f.assign_columns(1, g)
        # Captured continuing native seed6178 stream after source clauses1–12.
        with np.load(_FIXTURE.with_name("assign_columns_native_queries.npz")) as data:
            x = jnp.asarray(data["x100_k1"])
        err = float(np.linalg.norm(np.asarray(h(x)) - np.asarray(oph(x)), ord=np.inf))
        assert err < 1e2 * h.vscale * EPS
