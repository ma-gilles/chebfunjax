"""Dense rectangular basic QR against independent MATLAB R2025b captures.

Supplemental new controls, not borrowed Chebfun acceptance predicates.
Exact rank/zero-variable checks plus a fixed 32-eps comparison for the small
well-scaled solutions (including scaled diagonal cases) are set before runs.
"""
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import qr

from chebfunjax.utils._basic_lstsq import _basic_lstsq, _basic_qr_rank

DATA = Path(__file__).parent / "data"
RECORDS = json.loads((DATA / "basic_lstsq_native_r2025b.json").read_text())["records"]

@pytest.mark.parametrize("record", RECORDS, ids=[r["label"] for r in RECORDS])
def test_native_rectangular_controls(record):
    with np.load(DATA / "basic_lstsq_native_r2025b.npz") as fixture:
        i = record["index"]
        a, b, expected, pivots = [fixture[f"{i}_{key}"] for key in ["a", "b", "x", "p"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = np.asarray(_basic_lstsq(a, b))
    rank = record["rank"]
    assert bool(caught) == (rank < min(a.shape))
    if caught:
        assert f"rank = {rank}," in str(caught[0].message)
    _, r, _ = qr(jnp.asarray(a, dtype=jnp.complex128 if np.iscomplexobj(a) else jnp.float64), mode="economic", pivoting=True)
    assert _basic_qr_rank(r, *a.shape)[0] == rank
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual[np.asarray(pivots, dtype=int).ravel()[rank:] - 1], 0)
    np.testing.assert_allclose(actual, expected, rtol=32*np.finfo(float).eps, atol=32*np.finfo(float).eps)

@pytest.mark.parametrize("a,b", [(jnp.eye(2),jnp.ones(2)), (jnp.ones((0,2)),jnp.ones(0)), (jnp.ones((3,2)),jnp.ones(2)), (jnp.ones(3),jnp.ones(3))])
def test_scope_validation(a,b):
    with pytest.raises(ValueError):
        _basic_lstsq(a,b)
