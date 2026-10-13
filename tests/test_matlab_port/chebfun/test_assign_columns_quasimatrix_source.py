"""Native genuine-Quasimatrix clauses for ``test_assignColumns``.

Provenance
----------
MATLAB source : tests/chebfun/test_assignColumns.m, clauses 1--15
Chebfun commit: 7574c77
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain

EPS = float(np.finfo(np.float64).eps)
_CAPTURE = np.load(Path(__file__).parent / "fixtures" /
                   "assign_columns_native_queries.npz")
X = jnp.asarray(_CAPTURE["x10_k2"])
DOM = Domain((-1.0, 0.0, 1.0))


def _q(functions, domain=DOM):
    """Mirror native quasimatrix(op, domain): chebfun then cheb2quasi."""
    functions = tuple(functions)
    array_fun = cj.chebfun(
        lambda x, _functions=functions: jnp.stack(
            [function(x) for function in _functions], axis=-1),
        domain=tuple(domain.breakpoints))
    return Quasimatrix(
        [array_fun.extract_columns(k) for k in range(array_fun.n_columns)],
        array_fun.domain)


def _assert_source_error(actual, exact):
    # MATLAB norm(err(:),inf); Quasimatrix.vscale is per-column in Python,
    # so take its max to adapt MATLAB's scalar vscale(F).
    error = jnp.max(jnp.abs(jnp.ravel(actual(X) - exact)))
    scale = jnp.max(actual.vscale())
    assert float(error) < 10.0 * float(scale) * EPS


def _f():
    return _q([jnp.sin, jnp.cos, jnp.exp])

def test_quasimatrix_assign_columns_clauses_1_to_5():
    f = _f()
    coarse = Domain((-1.0, 1.0))
    g = _q([lambda x: x], coarse)
    h = f.assign_columns(0, g)
    _assert_source_error(h, jnp.stack((X, jnp.cos(X), jnp.exp(X)), axis=-1))

    h = f.assign_columns(2, g)
    _assert_source_error(h, jnp.stack((jnp.sin(X), jnp.cos(X), X), axis=-1))

    g = _q([lambda x: x, lambda x: x**2])
    h = f.assign_columns([2, 0], g)
    _assert_source_error(h, jnp.stack((X**2, jnp.cos(X), X), axis=-1))

    h = f.assign_columns([1, 1], g)
    _assert_source_error(h, jnp.stack((jnp.sin(X), X**2, jnp.exp(X)), axis=-1))

    g = _q([lambda x: x, lambda x: x**2, lambda x: x**3],
           Domain((-1.0, -0.5, 0.5, 1.0)))
    h = f.assign_columns([0, 1, 2], g)
    assert isinstance(h, Quasimatrix)
    _assert_source_error(h, jnp.stack((X, X**2, X**3), axis=-1))


def test_quasimatrix_assign_columns_clauses_6_to_8():
    f = _f()
    g = _q([lambda x: x, lambda x: x**2, lambda x: x**3],
           Domain((-1.0, -0.5, 0.5, 1.0)))
    h = f.assign_columns(":", g)
    hc = f.assign_columns([0, 1, 2], g)
    assert h.isequal(hc)

    h = f.assign_columns([1, 0], [-0.5, 0.5])
    _assert_source_error(
        h, jnp.stack((jnp.full_like(X, 0.5), jnp.full_like(X, -0.5),
                      jnp.exp(X)), axis=-1))

    row = f.T.assign_columns([1, 0], jnp.array([[-0.5], [0.5]]))
    assert row.is_transposed
    expected = jnp.stack((jnp.full_like(X, 0.5), jnp.full_like(X, -0.5),
                          jnp.exp(X)), axis=-1).T
    error = jnp.max(jnp.abs(jnp.ravel(row(X) - expected)))
    assert float(error) < 10.0 * float(jnp.max(row.vscale())) * EPS


def test_quasimatrix_assign_columns_errors_and_growth_clauses_9_to_12():
    f = _f()
    g = _q([lambda x: x, lambda x: x**2, lambda x: x**3],
           Domain((-1.0, -0.5, 0.5, 1.0)))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:assignColumns:numCols"):
        f.assign_columns([0, 1, 2], g.T)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:assignColumns:numCols"):
        f.assign_columns([0, 1], g)
    g2 = _q([lambda x: x, lambda x: x**2, lambda x: x**3],
            Domain((0.0, 1.0)))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:assignColumns:domain"):
        f.assign_columns([0, 1, 2], g2)

    h = f.assign_columns([0, 1, 3], g)
    assert len(h.cols) == 4
    _assert_source_error(h, jnp.stack((X, X**2, jnp.exp(X), X**3), axis=-1))

    with pytest.raises(IndexError, match="column index out of range"):
        f.assign_columns(4, None)
    with pytest.raises(TypeError, match="booleans"):
        f.assign_columns(True, g)
    with pytest.raises(TypeError, match="booleans"):
        f.assign_columns(jnp.asarray([True, False]), g)

    # Row Quasimatrix growth pads with a row-oriented zero column, retaining
    # its orientation and source column order.
    row = f.T.assign_columns([1, 4], jnp.asarray([2.5, 4.5]))
    assert isinstance(row, Quasimatrix)
    assert row.is_transposed and len(row.cols) == 5
    row_expected = jnp.stack((jnp.sin(X), jnp.full_like(X, 2.5),
                              jnp.exp(X), jnp.zeros_like(X),
                              jnp.full_like(X, 4.5)), axis=0)
    row_error = jnp.max(jnp.abs(jnp.ravel(row(X) - row_expected)))
    assert float(row_error) < 10.0 * float(jnp.max(row.vscale())) * EPS


def test_quasimatrix_remove_column_clauses_14_and_15():
    x = cj.chebfun("x")
    # Native pass14 forms an array-valued [x, 2*x] and applies cheb2quasi.
    dense = cj.Chebfun.horzcat(x, 2 * x)
    xx = Quasimatrix(dense.mat2cell(), dense.domain)
    reduced = xx.assign_columns(1, None)
    assert len(reduced.cols) == 1
    assert float((x - reduced).norm()) < EPS

    # pass(k,15): source constructs [x, 2*abs(x)] as a Quasimatrix,
    # then removes the second column.
    xx = Quasimatrix([x, 2 * abs(x)], x.domain)
    reduced = xx.assign_columns(1, None)
    assert len(reduced.cols) == 1
    assert float((x - reduced).norm()) < EPS

    # Native pass15's different breakpoints make [x, 2*abs(x)] a
    # Quasimatrix directly; retain that source concatenation route.
    xx = cj.Chebfun.horzcat(x, 2 * abs(x))
    assert isinstance(xx, Quasimatrix)
    reduced = xx.assign_columns(1, None)
    assert len(reduced.cols) == 1
    assert float((x - reduced).norm()) < EPS


def test_quasimatrix_unbounded_clause_13_literal_continuation_draws():
    # Native pass13 samples after executing the complete clauses1--12 prefix.
    domain = Domain((-jnp.inf, -3.0 * jnp.pi))
    f = _q([jnp.exp, lambda x: x * jnp.exp(x),
            lambda x: (1.0 - jnp.exp(x)) / x], domain)
    g = _q([lambda x: jnp.exp(-(x**2))], domain)
    h = f.assign_columns(1, g)

    x = jnp.asarray(_CAPTURE["x100_k2"])
    expected = jnp.stack((jnp.exp(x), jnp.exp(-(x**2)),
                          (1.0 - jnp.exp(x)) / x), axis=-1)
    error = h(x) - expected
    # MATLAB norm(err,inf) on this 100x3 matrix is the maximum row sum.
    source_norm = jnp.max(jnp.sum(jnp.abs(error), axis=1))
    source_vscale = jnp.max(h.vscale())
    assert float(source_norm) < 100.0 * float(source_vscale) * EPS
