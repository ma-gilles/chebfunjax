"""Implicitly differentiated solve with source real binary64 LU factors.

Provenance
----------
MATLAB source: @valsDiscretization/mldivide.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
The original matrix defines differentiation. Row scaling belongs inside the
forward and transpose callbacks, keeping scale-selection ties out of the AD.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._source_lu_factor import general_factor


@jax.jit
def _factors(a):
    lu, pivots, _certificate = general_factor(a)

    def swap(i, permutation):
        first, second = permutation[i], permutation[pivots[i]]
        return permutation.at[i].set(second).at[pivots[i]].set(first)

    permutation = jax.lax.fori_loop(0, a.shape[0], swap, jnp.arange(a.shape[0]))
    return lu, permutation


def _apply(lu, permutation, rhs, transpose=False):
    values = rhs[:, None] if rhs.ndim == 1 else rhs
    if transpose:
        values = jax.lax.linalg.triangular_solve(
            lu, values, left_side=True, lower=False, transpose_a=True
        )
        values = jax.lax.linalg.triangular_solve(
            lu, values, left_side=True, lower=True, unit_diagonal=True, transpose_a=True
        )
        values = values[jnp.argsort(permutation)]
    else:
        values = jax.lax.linalg.triangular_solve(
            lu, values[permutation], left_side=True, lower=True, unit_diagonal=True
        )
        values = jax.lax.linalg.triangular_solve(lu, values, left_side=True, lower=False)
    return values[:, 0] if rhs.ndim == 1 else values


def solve_with_row_scaling(matrix, rhs, scaled_matrix, scale):
    """Solve with original-A implicit derivatives and vector/matrix RHS.

    Caller supplies the source-scaled matrix and scalar-per-row multiplier.
    This helper handles real binary64 arrays; the caller retains its existing
    JAX solve for other dtypes. No primal/tangent input admission gate is used.
    """
    lu, permutation = _factors(jax.lax.stop_gradient(scaled_matrix))
    scale = jax.lax.stop_gradient(scale)
    multiplier = scale if rhs.ndim == 1 else scale[:, None]
    return jax.lax.custom_linear_solve(
        lambda x: matrix @ x,
        rhs,
        solve=lambda _, b: _apply(lu, permutation, multiplier * b),
        transpose_solve=lambda _, b: multiplier * _apply(lu, permutation, b, True),
    )
