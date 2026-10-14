"""Source-ordered rank-one updates for disk Phase II.

Provenance
----------
Chebfun pin7574c77680d7e82b79626300bf255498271a72df,
@diskfun/constructor.m, PhaseTwo. The source computes a reciprocal, scales
selected slice entries, forms an outer product, then subtracts it. CPU
binary64 qualification uses retained MATLAB R2025b nine-stage inputs.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary


def _subtract_outer(matrix, left, right):
    def step(index, state):
        residual, product = state
        return jax.lax.cond(
            index == 0,
            lambda: (residual, left[:, None] * right[None, :]),
            lambda: (residual - product, product),
        )

    # Different iterations preserve source multiply/subtract rounding without
    # constructing the full-matrix operand stacks used by binary().
    return jax.lax.fori_loop(0, 2, step, (matrix, jnp.zeros_like(matrix)))[0]


@jax.jit
def phase_two_rank_one(cols, rows, col, row, pivot, row_indices, col_indices):
    """Update one plus or minus component without changing pivot selection.

    Source rounding is qualified on CPU float64. Other dtypes/backends use
    the same source expression order with ordinary JAX arithmetic. No claim
    of cross-backend bit identity is made.
    """
    def generic(cols, rows, col, row, pivot, row_indices, col_indices):
        reciprocal = 1 / pivot
        return (
            cols - col[:, None] * (row[col_indices] * reciprocal)[None, :],
            rows - (reciprocal * col[row_indices])[:, None] * row[None, :],
        )

    def ordered(cols, rows, col, row, pivot, row_indices, col_indices):
        reciprocal = binary(jnp.ones_like(pivot), pivot, 3)
        scaled_row = binary(row[col_indices], reciprocal, 0)
        scaled_col = binary(reciprocal, col[row_indices], 0)
        return (
            _subtract_outer(cols, col, scaled_row),
            _subtract_outer(rows, scaled_col, row),
        )

    args = (cols, rows, col, row, pivot, row_indices, col_indices)
    if any(value.dtype != jnp.float64 for value in args[:5]):
        return generic(*args)
    return jax.lax.platform_dependent(*args, cpu=ordered, default=generic)
