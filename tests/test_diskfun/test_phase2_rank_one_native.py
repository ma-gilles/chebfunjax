"""CPU native rank-one residuals captured before the production correction.

Provenance: @diskfun/constructor.m PhaseTwo, Chebfun pin7574c77.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._rank_one import phase_two_rank_one

FIXTURE = Path(__file__).parents[1] / "references/disk_phase2_rank_one_native.npz"


@pytest.mark.parametrize("outer_jit", [False, True])
def test_native_nine_stage_residuals(outer_jit):
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        pytest.skip("Native residual-bit qualification is CPU binary64")
    update = jax.jit(phase_two_rank_one) if outer_jit else phase_two_rank_one
    with np.load(FIXTURE) as data:
        cols = jnp.asarray(data["new_cols_plus"])
        rows = jnp.asarray(data["new_rows_plus"])
        for i, pivot in enumerate(data["pivot_array"][:, 0]):
            cols, rows = update(
                cols, rows, cols[:, i], rows[i, :], pivot,
                data["id_rows_cur"], data["id_cols_cur"],
            )
            for name, actual in (("newColsPlus", cols), ("newRowsPlus", rows)):
                np.testing.assert_array_equal(
                    np.asarray(actual).view(np.uint64),
                    data[f"{name}_{i}"].view(np.uint64),
                )
