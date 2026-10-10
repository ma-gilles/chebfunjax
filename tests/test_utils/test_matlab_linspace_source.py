"""Eight exact native R2025b Needle grids, 500 binary64 coordinates."""

import base64
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._matlab_linspace import source_grid

NATIVE = Path(__file__).resolve().parents[1] / "fixtures" / "needle_native_coordinates_r2025b.json"


@pytest.mark.parametrize("grid_index", [0, 1])
@pytest.mark.parametrize("axis", ["x", "theta", "xxp", "ttp"])
def test_native_coordinates(grid_index, axis):
    item = json.loads(NATIVE.read_text())["grids"][grid_index][axis]
    raw = base64.b64decode(item["raw_storage_base64"])
    native = np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"], order="F")
    expected = native[:, 0] if axis == "ttp" else native[0, :]
    actual = np.asarray(source_grid(float(expected[0]), float(expected[-1]), expected.size))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert bool(jnp.all(jnp.isfinite(actual)))
