"""High-rank bit-identity regression for fixed-shape skeleton updates."""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.separable_approx import _ge_on_skeleton


def test_skeleton_fixed_shape_rank40_matches_d478_helper_bitwise():
    rng = np.random.RandomState(1740)
    rank, ny, nx = 40, 23, 31
    columns = rng.standard_normal((ny, rank))
    rows = rng.standard_normal((rank, nx))
    pivots = rng.standard_normal(rank) + 1.25
    pivots[np.abs(pivots) < 0.1] += 0.5
    positions = np.column_stack(
        [rng.randint(0, ny, size=rank), rng.randint(0, nx, size=rank)]
    )

    expected_columns = jnp.asarray(columns)
    expected_rows = jnp.asarray(rows)
    pivot_values = jnp.asarray(pivots)
    pivot_positions = jnp.asarray(positions)
    for k in range(rank - 1):
        pivot = pivot_values[k]
        row_at_pivot_y = pivot_positions[k + 1 :, 0]
        col_at_pivot_x = pivot_positions[k + 1 :, 1]
        numerator = expected_rows[k, col_at_pivot_x]
        scale = numerator / jnp.full_like(numerator, pivot)
        expected_columns = expected_columns.at[:, k + 1 :].set(
            expected_columns[:, k + 1 :]
            - jnp.outer(expected_columns[:, k], scale)
        )
        numerator = expected_rows[k, :]
        row_scale = numerator / jnp.full_like(numerator, pivot)
        expected_rows = expected_rows.at[k + 1 :, :].set(
            expected_rows[k + 1 :, :]
            - jnp.outer(expected_columns[row_at_pivot_y, k], row_scale)
        )

    actual_columns, actual_rows = _ge_on_skeleton(
        columns, rows, pivots, positions
    )
    np.testing.assert_array_equal(actual_columns, expected_columns)
    np.testing.assert_array_equal(actual_rows, expected_rows)
