"""Literal Singfun clauses 7--9 of mldivide and 6--8 of mrdivide.

MATLAB source : tests/chebfun/test_mldivide.m and test_mrdivide.m
Chebfun commit: 7574c77
Native sample inputs use the consecutive R2025b seedRNG(6178) fixture.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import (
    _mrdivide_adjoint,
    mldivide,
    mrdivide,
)

EPS = float(jnp.finfo(jnp.float64).eps)
XR = jnp.asarray(json.loads(
    (Path(__file__).parent / 'fixtures' / 'abs_seed6178_native.json').read_text()
)['x']).reshape((-1, 1))


def _singfun(op, exponents):
    return cj.chebfun(op, exps=exponents, splitting=True)


def _mldivide7_f():
    return _singfun(lambda x: jnp.sin(20.0 * x) / (x + 1.0), (-1.0, 0.0)).T


def _mldivide8_f():
    return _singfun(
        lambda x: (jnp.sin(100.0 * x) ** 2 + 1.0) / (x + 1.0) ** 0.25,
        (-0.25, 0.0),
    ).T


def _column_ratio_pair():
    f = _singfun(lambda x: 3.0 * (x**2 + 3.0) / (x + 1.0) ** 0.4,
                 (-0.4, 0.0))
    g = _singfun(lambda x: (x**2 + 3.0) / (x + 1.0) ** 0.4,
                 (-0.4, 0.0))
    return f, g


def _mrdivide7_f():
    return _singfun(
        lambda x: (jnp.sin(100.0 * x) ** 2 + 1.0) / (1.0 - x) ** 0.4,
        (0.0, -0.4),
    )


def _mrdivide8_pair():
    f = _singfun(
        lambda x: (jnp.sin(30.0 * x) ** 2 + 1.0) / (1.0 - x) ** 0.3,
        (0.0, -0.3),
    ).T
    g = _singfun(
        lambda x: 9.0 * (jnp.sin(30.0 * x) ** 2 + 1.0) / (1.0 - x) ** 0.3,
        (0.0, -0.3),
    ).T
    return f, g


def test_mldivide_source07_scalar_over_row_singfun():
    f = _mldivide7_f()
    g = mldivide(3.0, f)
    values = g(XR)
    # The source explicitly transposes op(xr), matching row-Chebfun feval.
    expected = (jnp.sin(20.0 * XR) / (3.0 * (XR + 1.0))).T
    assert jnp.max(jnp.abs(values - expected)) < 1e5 * g.vscale * EPS


def test_mldivide_source08_row_singfun_over_scalar_ls():
    f = _mldivide8_f()
    g = mldivide(f, 3.0)
    # The source f*g is a row-by-column scalar product. In this Python API,
    # use the equivalent continuous inner product rather than pointwise *.
    error = jnp.abs(f.T.inner(g) - 3.0)
    assert error < 10.0 * g.vscale * EPS


def test_mldivide_source09_column_singfun_over_column_singfun_ls():
    f, g = _column_ratio_pair()
    h = mldivide(f, g)
    error = jnp.abs(h - 1.0 / 3.0)
    assert error < f.vscale * EPS


def test_mrdivide_source06_column_singfun_over_scalar():
    f = _singfun(lambda x: jnp.sin(20.0 * x) / (x + 1.0), (-1.0, 0.0))
    g = mrdivide(f, 3.0)
    values = g(XR)
    expected = jnp.sin(20.0 * XR) / (3.0 * (XR + 1.0))
    assert jnp.max(jnp.abs(values - expected)) < 1e5 * g.vscale * EPS


def test_mrdivide_source07_scalar_over_column_singfun_ls():
    f = _mrdivide7_f()
    g = mrdivide(5.0, f)
    # Native g*f is row-by-column; map it to the Python inner product.
    error = jnp.abs(g.T.inner(f) - 5.0)
    assert error < 100.0 * g.vscale * EPS


def test_mrdivide_source08_row_singfun_over_row_singfun_ls():
    f, g = _mrdivide8_pair()
    h = mrdivide(g, f)
    error = jnp.abs(h - 9.0)
    assert error < 100.0 * f.vscale * EPS


def test_mldivide_complex_row_uses_conjugate_transpose():
    # MATLAB A\B conjugates a complex row A when entering the QR branch.
    # This control distinguishes source A' from the nonconjugating A.'.
    c = 1.0 + 2.0j
    rhs = 2.0 + 3.0j
    row = cj.chebfun(lambda x: c + 0.0 * x).T
    expected = jnp.conj(c) * rhs / (2.0 * jnp.abs(c) ** 2)
    left_solution = mldivide(row, rhs)
    right_solution = mrdivide(rhs, row.T)
    query = jnp.asarray([-0.5, 0.0, 0.5])
    assert jnp.max(jnp.abs(left_solution(query) - expected)) < 20 * EPS
    assert jnp.max(jnp.abs(right_solution(query) - expected)) < 20 * EPS


def test_mrdivide_numeric_matrix_adjoint_and_dimension_error():
    matrix = jnp.asarray([[1.0 + 2.0j, 3.0 - 1.0j],
                          [-2.0j, 4.0 + 5.0j]])
    adjoint = _mrdivide_adjoint(matrix)
    assert adjoint.shape == (2, 2)
    assert jnp.array_equal(adjoint, jnp.conj(matrix.T))
    scalar_adjoint = _mrdivide_adjoint(1.0 + 2.0j)
    assert scalar_adjoint.shape == ()
    assert scalar_adjoint == 1.0 - 2.0j

    from chebfunjax.chebfun1d.linalg import Quasimatrix

    col = cj.chebfun(lambda x: 1.0 + x)
    quasi_adjoint = _mrdivide_adjoint(Quasimatrix([col], col.domain))
    assert quasi_adjoint.is_transposed
    assert quasi_adjoint.cols[0].is_transposed

    # Native mrdivide checks the matrix dimensions before entering its
    # numeric-denominator branch.
    f = cj.chebfun(lambda x: 1.0 + x)
    with pytest.raises(
            ValueError,
            match=r"CHEBFUN:CHEBFUN:mrdivide:dimensions"):
        mrdivide(f, matrix)
