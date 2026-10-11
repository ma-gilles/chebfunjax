"""Retained source-pipeline regression, qualified on JAX 0.11 CPU binary64."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._simplify import _cheb_cutoff, _trig_cutoff, simplify_slice
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

_REF = Path(__file__).parents[1] / "references" / "disk_simplify_retained_cpu"
_CASES = json.loads(_REF.with_suffix(".json").read_text())["cases"]
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.x64_enabled,
    reason="Retained source pipeline qualified on CPU binary64",
)


@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"{c['kind']}_{c['index']}")
def test_retained_noisy_cutoff_and_simplified_coefficients(case):
    """Raw cutoffs also detect bypassing the noisy round-trip before capping."""
    key = f"{case['kind']}_{case['index']}"
    with np.load(_REF.with_suffix(".npz")) as data:
        coeffs = jnp.asarray(data[key + "_input"])
        expected = data[key + "_expected"]
    meta = case["tech_metadata"]
    if case["kind"] == "cols":
        tech = Chebtech2.from_coeffs(coeffs, ishappy=meta["ishappy"])
        cutoff = _cheb_cutoff
    else:
        tech = Trigtech.from_coeffs(
            coeffs,
            ishappy=meta["ishappy"],
            real_columns=tuple(meta["real_columns"]),
        )
        cutoff = _trig_cutoff
    assert int(cutoff(coeffs, jnp.asarray(case["tol"]))) == case["baseline_cutoff"]
    actual = simplify_slice(tech, case["tol"])
    result = np.asarray(actual.coeffs)
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert result.tobytes() == expected.tobytes()
    assert actual.ishappy == meta["ishappy"]
    if case["kind"] == "rows":
        assert actual.is_real
        assert actual.real_columns == tuple(meta["real_columns"])
