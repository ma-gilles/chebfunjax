"""Retained local scales and stored-value semantics for disk CPU fusion."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._simplify import slice_scale
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

_REF = Path(__file__).parents[1] / "references"
_CASES = json.loads((_REF / "disk_scale_retained_cpu.json").read_text())["cases"]
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.x64_enabled,
    reason="Retained CPU binary64 scale qualification",
)


@pytest.mark.parametrize("case", _CASES, ids=lambda c: f"{c['kind']}_{c['index']}")
def test_retained_scale(case):
    with np.load(_REF / "disk_simplify_retained_cpu.npz") as data:
        coeffs = jnp.asarray(data[f"{case['kind']}_{case['index']}_input"])
    meta = case["metadata"]
    if case["kind"] == "cols":
        tech = Chebtech2.from_coeffs(coeffs, ishappy=meta["ishappy"])
    else:
        tech = Trigtech.from_coeffs(
            coeffs, ishappy=meta["ishappy"], real_columns=tuple(meta["real_columns"])
        )
    assert slice_scale(tech) == case["scale"]


def test_stored_trig_values_remain_authoritative():
    tech = Trigtech(
        coeffs=jnp.asarray([0.0, 1.0, 0.0]),
        is_real=True,
        real_columns=(True,),
        ishappy=True,
        _values=jnp.asarray([-3.0, 4.0, -2.0]),
    )
    assert slice_scale(tech) == 4.0
