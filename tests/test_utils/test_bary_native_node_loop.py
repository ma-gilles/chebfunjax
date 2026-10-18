"""Native real node-loop arithmetic, including compiler rounding boundaries.

Provenance: Chebfun 7574c77 bary.m node-loop branch, called by the local
FUNQUI body of @smoothfun/smoothfun.m. Fixture driver/pins and limitations
are in tests/fixtures/bary_node_loop_native_20261017.json.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.interpolation import bary


@pytest.mark.parametrize("n", [65, 129, 257])
def test_cpu_real_node_loop_matches_captured_native_words(n):
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        pytest.skip("Captured ordinary CPU binary64 operation contract")
    fixture = Path(__file__).parents[1]/"fixtures/bary_node_loop_native_20261017.npz"
    with np.load(fixture, allow_pickle=False) as data:
        actual = np.asarray(bary(jnp.asarray(data[f"points_{n}"]),
                                 jnp.asarray(data["v"]), jnp.asarray(data["nodes"]),
                                 jnp.asarray(data["weights"])))
        expected = data[f"values_{n}"]
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
