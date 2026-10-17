"""Independent primal/implicit AD controls for owned normest triangular actions."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import pytest

from chebfunjax.utils._normest_triangular import triangular_solve


@pytest.mark.parametrize(
    "lower,transpose,unit",
    [(True, False, True), (False, False, False), (False, True, False), (True, True, True)],
)
@pytest.mark.parametrize("zero", [False, True])
def test_primal_jvp_vjp_independent_zero_rhs_and_mask(lower, transpose, unit, zero):
    n = 6
    a = np.arange(n * n, dtype=float).reshape(n, n) / 100 + np.diag(np.arange(n) + 2.0)
    b = np.zeros((n, 3)) if zero else (np.arange(n * 3, dtype=float).reshape(n, 3) - 7) / 11
    da = np.cos(np.arange(n * n)).reshape(n, n) / 13
    db = np.sin(np.arange(n * 3)).reshape(n, 3) / 7
    mask = np.tril(np.ones_like(a)) if lower else np.triu(np.ones_like(a))
    aa = a * mask
    daa = da * mask
    if unit:
        np.fill_diagonal(aa, 1.0)
        np.fill_diagonal(daa, 0.0)
    if transpose:
        aa = aa.T
        daa = daa.T
    expected = np.linalg.solve(aa, b)

    def fn(a, b):
        return triangular_solve(a, b, lower=lower, transpose=transpose, unit=unit)

    actual, tangent = jax.jvp(
        fn, (jnp.asarray(a), jnp.asarray(b)), (jnp.asarray(da), jnp.asarray(db))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        tangent, np.linalg.solve(aa, db - daa @ expected), rtol=1e-12, atol=1e-13
    )
    weight = np.arange(n * 3, dtype=float).reshape(n, 3) / 17 - 0.5
    _, pull = jax.vjp(fn, jnp.asarray(a), jnp.asarray(b))
    abar, bbar = pull(jnp.asarray(weight))
    expected_bbar = np.linalg.solve(aa.T, weight)
    expected_abar = -(expected_bbar @ expected.T)
    if transpose:
        expected_abar = expected_abar.T
    expected_abar *= mask
    if unit:
        np.fill_diagonal(expected_abar, 0.0)
    np.testing.assert_allclose(bbar, expected_bbar, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(abar, expected_abar, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("n,dtype", [(11, jnp.float64), (6, jnp.float32), (6, jnp.complex128)])
def test_outside_scope_generic(n, dtype):
    a = jnp.eye(n, dtype=dtype) * 2
    b = jnp.arange(n, dtype=jnp.float64).astype(dtype)
    np.testing.assert_array_equal(
        triangular_solve(a, b, lower=False), jl.solve_triangular(a, b, lower=False)
    )


def test_vector_rhs_and_ignored_unit_triangle():
    a = jnp.array([[99.0, jnp.nan, jnp.nan], [0.25, 88.0, jnp.nan], [-0.5, 0.75, 77.0]])
    b = jnp.array([1.0, 2.0, 3.0])
    expected = np.linalg.solve(
        np.array([[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [-0.5, 0.75, 1.0]]), np.asarray(b)
    )
    np.testing.assert_allclose(
        triangular_solve(a, b, lower=True, unit=True), expected, rtol=1e-12, atol=1e-13
    )
