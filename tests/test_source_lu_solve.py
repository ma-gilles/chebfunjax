"""Independent solve/derivative contracts for source LU integration.

Provenance
----------
MATLAB source: @valsDiscretization/mldivide.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
These independent linear-algebra controls supplement unchanged native LU tests.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import _piecewise_row_scaled_solve as solve
from chebfunjax.utils._source_lu_solve import _factors


def _assert_close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("n", [3, 17])
@pytest.mark.parametrize("columns", [1, 3])
def test_matrix_rhs_primal_jit_jvp_and_vjp(n, columns):
    i = np.arange(1, n+1, dtype=float)
    a = np.diag(i+3) + 0.05*np.sin(i[:, None]*i[None, :])
    b = np.cos(i[:, None] + np.arange(columns)[None, :])
    h = 0.02*np.cos(i[:, None] - i[None, :])
    db = 0.03*np.sin(b)
    g = np.arange(n*columns, dtype=float).reshape(n, columns)/(n*columns)+0.2
    if columns == 1:
        b, db, g = b[:, 0], db[:, 0], g[:, 0]
    x = np.linalg.solve(a, b)
    aa, bb = jnp.asarray(a), jnp.asarray(b)
    _assert_close(solve(aa, bb), x)
    _assert_close(jax.jit(solve)(aa, bb), x)
    primal, tangent = jax.jvp(solve, (aa, bb), (jnp.asarray(h), jnp.asarray(db)))
    _assert_close(primal, x)
    _assert_close(tangent, np.linalg.solve(a, db-h@x))
    _, pullback = jax.vjp(solve, aa, bb)
    da, d_b = pullback(jnp.asarray(g))
    adjoint = np.linalg.solve(a.T, g)
    expected_da = -np.outer(adjoint, x) if columns == 1 else -adjoint@x.T
    _assert_close(da, expected_da)
    _assert_close(d_b, adjoint)


def test_scale_ties_and_second_derivative_use_original_matrix():
    a = np.array([[2., -2.], [1., 3.]])
    h = np.array([[.3, -.1], [.2, .4]])
    b = np.array([1., 4.])
    def loss(t):
        return jnp.sum(solve(jnp.asarray(a)+t*jnp.asarray(h), jnp.asarray(b)))
    x = np.linalg.solve(a, b)
    expected = np.sum(2*np.linalg.solve(a, h@np.linalg.solve(a, h@x)))
    _assert_close(jax.grad(jax.grad(loss))(jnp.asarray(0.)), expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64, jnp.complex128])
def test_other_dtypes_retain_jax_solve(dtype):
    a = jnp.asarray([[3., .2], [-.1, 2.]], dtype=dtype)
    b = jnp.asarray([[1., 2.], [3., 4.]], dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        a = a + 0.1j*jnp.eye(2, dtype=dtype)
        b = b + 0.3j
    actual = solve(a, b)
    assert actual.dtype == jnp.dtype(dtype)
    expected = np.linalg.solve(np.asarray(a), np.asarray(b))
    tol = 20*np.finfo(np.asarray(actual).real.dtype).eps
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=tol, atol=tol)


def test_empty_matrix_rhs_and_vmap():
    a = jnp.array([[3., .2], [-.1, 2.]])
    assert solve(a, jnp.zeros((2, 0))).shape == (2, 0)
    matrices = jnp.stack((a, 2*a))
    rhs = jnp.array([[1., 3.], [2., 4.]])
    expected = np.stack([np.linalg.solve(np.asarray(x), np.asarray(b))
                         for x, b in zip(matrices, rhs)])
    _assert_close(jax.vmap(solve)(matrices, rhs), expected)


def test_nonsymmetric_row_permutation_matrix_rhs_primal_jvp_vjp():
    n, columns = 17, 3
    i = np.arange(1, n + 1, dtype=float)
    # Strong diagonal plus one-way coupling: nonsymmetry is deliberate.
    unpermuted = np.diag(i + 3) + 0.2 * np.diag(np.ones(n - 1), 1)
    order = np.roll(np.arange(n), 1)
    a = unpermuted[order, :]
    b = np.cos(i[:, None] + np.arange(columns)[None, :])
    h = 0.02 * np.sin(i[:, None] + 2 * i[None, :])
    db = 0.03 * np.sin(b)
    g = np.arange(n * columns, dtype=float).reshape(n, columns) / (n * columns) + 0.2
    aa, bb = jnp.asarray(a), jnp.asarray(b)

    # Assert the control really exercises row permutations after the same
    # max-row scaling convention used by the public piecewise solve.
    row_scale = 1.0 / np.max(np.abs(a), axis=1)
    _, permutation = _factors(jnp.asarray(a * row_scale[:, None]))
    assert not np.array_equal(np.asarray(permutation), np.arange(n))
    assert not np.array_equal(a, a.T)
    adjoint = np.linalg.solve(a.T, g)
    incorrect_forward = np.linalg.solve(a, g)
    assert not np.allclose(adjoint, incorrect_forward, rtol=1e-12, atol=1e-13)

    x = np.linalg.solve(a, b)
    _assert_close(solve(aa, bb), x)
    _assert_close(jax.jit(solve)(aa, bb), x)
    primal, tangent = jax.jvp(solve, (aa, bb), (jnp.asarray(h), jnp.asarray(db)))
    _assert_close(primal, x)
    _assert_close(tangent, np.linalg.solve(a, db - h @ x))

    _, pullback = jax.vjp(solve, aa, bb)
    da, d_b = pullback(jnp.asarray(g))
    _assert_close(da, -adjoint @ x.T)
    _assert_close(d_b, adjoint)
