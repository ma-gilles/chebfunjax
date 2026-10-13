"""Focused native assignment predicates absent from the mapped test module.

Provenance
----------
MATLAB source : tests/chebfun/test_assignColumns.m, @chebfun/assignColumns.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix, _source_quasi2cheb
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


def test_native_pass15_delete_piecewise_abs_column():
    # Native pass(1,15): xx = quasi2cheb([x, 2*abs(x)]); xx(:,2) = [].
    x = cj.chebfun(lambda t: t)
    raw = cj.Chebfun.horzcat(x, 2 * abs(x))
    assert isinstance(raw, Quasimatrix)
    # The source helper performs its own domain union, restriction, and
    # orientation-aware concatenation.
    xx = _source_quasi2cheb(raw.cols)
    assert isinstance(xx, cj.Chebfun)
    reduced = xx.assign_columns(1, None)
    assert reduced.n_columns == 1
    assert float((x - reduced).norm()) < np.finfo(np.float64).eps


def test_numeric_complex_constant_assignment_preserves_imaginary_part():
    # Source @chebfun/assignColumns.m treats numeric g as a Chebfun constant.
    f = cj.chebfun(lambda t: jnp.stack([t, 2 * t], axis=-1))
    assigned = f.assign_columns(0, 1.0 + 2.0j)
    query = jnp.array([-0.5, 0.25])
    expected = jnp.stack([
        jnp.full(query.shape, 1.0 + 2.0j), 2 * query.astype(jnp.complex128)
    ], axis=-1)
    assert jnp.array_equal(assigned(query), expected)
    assert jnp.array_equal(
        assigned.point_values[:, 0], jnp.array([1.0 + 2.0j, 1.0 + 2.0j]))

    # Full-selector assignment returns the numeric constant Chebfun; preserve
    # row orientation and the complex vector in that early-return branch too.
    row_values = jnp.array([1.0 + 2.0j, 3.0 - 4.0j])
    row = f.T.assign_columns([0, 1], row_values)
    assert row.is_transposed
    expected_row = jnp.broadcast_to(row_values[:, None], (2, query.size))
    assert jnp.array_equal(row(query), expected_row)


def test_quasi2cheb_restricts_same_domain_periodic_columns():
    domain = (0.0, 2.0 * jnp.pi)
    periodic = cj.chebfun(
        lambda x: jnp.stack((jnp.cos(x), jnp.sin(x)), axis=-1),
        domain=domain, trig=True)
    q = Quasimatrix(periodic.mat2cell(), periodic.domain)

    assembled = _source_quasi2cheb(q.cols)
    assert isinstance(assembled, cj.Chebfun)
    assert all(isinstance(piece.tech, Chebtech2) for piece in assembled.funs)
    assert all(not isinstance(piece.tech, Trigtech) for piece in assembled.funs)

    # The same native conversion is used by one-column quasimatrix ./ vector.
    scalar_periodic = cj.chebfun(jnp.cos, domain=domain, trig=True)
    scalar_q = Quasimatrix(scalar_periodic.mat2cell(), scalar_periodic.domain)
    divided = scalar_q / jnp.array([1.0, 2.0])
    assert isinstance(divided, cj.Chebfun)
    assert all(isinstance(piece.tech, Chebtech2) for piece in divided.funs)
    query = jnp.array([0.2, 1.1, 2.4, 5.7])
    expected = jnp.stack((jnp.cos(query), jnp.cos(query) / 2.0), axis=-1)
    assert jnp.max(jnp.abs(divided(query) - expected)) < 1e-13
