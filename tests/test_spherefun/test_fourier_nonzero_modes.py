"""Independent dense source-equation checks for composed band solves."""

import numpy as np
import pytest

from chebfunjax.spherefun._fourier_bands import build_source_operators
from chebfunjax.spherefun._nonzero_modes import solve_nonzero_modes


@pytest.mark.parametrize("m,n", [(8, 7), (16, 12)])
@pytest.mark.parametrize("equation", ["poisson", "helmholtz"])
def test_source_mode_equations(m, n, equation):
    from chebfunjax.spherefun.spherefun import _sphere_fourier_operators

    d1, d2, dn, cs, s2, en, zero = _sphere_fourier_operators(m, n)
    ops = build_source_operators(m, n)
    f = (np.arange(m * n).reshape(m, n) + 1) / 17 + 1j * np.arange(m)[:, None] / 13
    laplace = s2 @ d2 + cs @ d1
    if equation == "poisson":
        operator = ops.laplace
        shifts = ops.dlambda2
        rhs, _, _ = ops.poisson_rhs(f)
        dense = laplace
        dense_shifts = np.diag(dn)
        corrected = f.copy()
        corrected[zero, n // 2] -= en @ f[:, n // 2] / en[zero]
        dense_rhs = s2 @ corrected
    else:
        k, c = 0.7 + 1e-14j, 1
        operator, shifts, rhs, _ = ops.helmholtz(f, k, c)
        dense = c * laplace / k**2 + s2
        dense_shifts = c * np.diag(dn) / k**2
        dense_rhs = s2 @ f / k**2
    indices, solution, singular, nonfinite = solve_nonzero_modes(
        operator.ab, shifts, rhs, lower=operator.lower, upper=operator.upper, zero_longitude=n // 2
    )
    indices, solution = np.asarray(indices), np.asarray(solution)
    assert np.array_equal(indices, np.r_[np.arange(n // 2 - 1, -1, -1), np.arange(n // 2 + 1, n)])
    assert not np.any(singular)
    assert not np.any(nonfinite)
    eps = np.finfo(float).eps
    for t, mode in enumerate(indices):
        a = dense + dense_shifts[mode] * np.eye(m)
        b = dense_rhs[:, mode]
        expected = np.linalg.solve(a, b)
        x = solution[:, t]
        scale = np.linalg.norm(a, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf)
        assert np.linalg.norm(a @ x - b, np.inf) <= 128 * m * eps * scale
        assert np.linalg.norm(x - expected, np.inf) <= (
            256 * m * eps * np.linalg.cond(a, np.inf) * max(1, np.linalg.norm(expected, np.inf))
        )


def test_exact_singular_and_nonfinite_flags():
    import jax.numpy as jnp

    # Three diagonal systems: singular zero shift, finite solution, Inf RHS.
    bands = jnp.zeros((3, 1))
    rhs = jnp.ones((3, 4)).at[1, 3].set(jnp.inf)
    indices, x, singular, nonfinite = solve_nonzero_modes(
        bands, jnp.array([0.0, 1.0, 2.0, 3.0]), rhs, lower=0, upper=0, zero_longitude=2
    )
    assert np.array_equal(indices, [1, 0, 3])
    assert np.array_equal(singular, [False, True, False])
    assert np.array_equal(nonfinite, [False, True, True])
    assert np.array_equal(np.asarray(x)[:, 0], np.ones(3))


def test_no_nonzero_longitudes():
    import jax.numpy as jnp

    indices, x, singular, nonfinite = solve_nonzero_modes(
        jnp.ones((3, 1)), jnp.zeros(1), jnp.ones((3, 1)), lower=0, upper=0, zero_longitude=0
    )
    assert indices.shape == singular.shape == nonfinite.shape == (0,)
    assert x.shape == (3, 0)
