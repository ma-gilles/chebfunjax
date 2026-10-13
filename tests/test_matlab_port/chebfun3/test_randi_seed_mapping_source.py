"""Chebfun3's private source-seeded fiber-index mapping."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import (
    Chebfun3,
    _initialize_random_indices,
)

jax.config.update("jax_enable_x64", True)
FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "matlab_chebfun3_randi_v1.json"
)
NATIVE = json.loads(FIXTURE.read_text())


def test_native_initialize_index_randomly_vectors():
    """Compare helper outputs to the captured literal MATLAB source vectors."""
    for case in NATIVE["cases"]:
        native = np.asarray(case["native_indices_1based"], dtype=int)
        actual = _initialize_random_indices(
            case["r"], case["maxVal"], case["restarts"]
        )
        if np.any(native > case["maxVal"]):
            # The literal source expression can exceed the grid for small r;
            # retain the existing Python clipping adapter in those cases.
            np.testing.assert_array_equal(
                actual, np.minimum(native, case["maxVal"]) - 1
            )
        else:
            np.testing.assert_array_equal(actual, native - 1)
        # Native J and K initialize independently but reseed to the same
        # restart seed, so the same input tuple must return the same indices.
        np.testing.assert_array_equal(
            _initialize_random_indices(
                case["r"], case["maxVal"], case["restarts"]
            ),
            actual,
        )


def test_source_seeded_indices_work_in_small_constructor():
    """Keep the helper wired into one bounded adaptive construction."""
    f = Chebfun3.from_function(
        lambda x, y, z: x + y + z,
        tol=1e-10,
        max_rank=8,
        min_samples=17,
        _restarts=1,
        _r_init=(3, 3, 3),
        _n_init=(17, 17, 17),
        _sample_test=False,
    )
    x = jnp.asarray([-0.7, 0.0, 0.6])
    y = jnp.asarray([0.1, -0.3, 0.4])
    z = jnp.asarray([0.2, 0.5, -0.8])
    actual = np.asarray(f(x, y, z))
    expected = np.asarray(x + y + z)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)
