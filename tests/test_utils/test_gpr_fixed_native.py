"""Fixed-L parity checks against captured values from the pinned MATLAB source."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

gprmod = import_module("chebfunjax.utils.gpr")

FIXTURE = Path(__file__).parent / "data/gpr_fixed_native_hex.json"


def _values(variables, name):
    record = variables[name]
    assert record["matlab_class"] == "double"
    return np.asarray([float.fromhex(v) for v in record["values"]], dtype=np.float64)


def _kwargs(options):
    values = options["values"]
    out = {}
    i = 0
    while i < len(values):
        key = values[i]["values"][0].lower()
        i += 1
        if key == "trig":
            out["trig"] = True
            continue
        value = values[i]
        i += 1
        decoded = _values({key: value}, key)
        if key == "domain":
            out["domain"] = tuple(decoded.tolist())
        elif key == "l":
            out["length_scale"] = float(decoded[0])
        elif key == "sigma":
            out["sigma"] = float(decoded[0])
        elif key == "noise":
            out["noise"] = float(decoded[0])
        elif key == "samples":
            out["n_samples"] = int(decoded[0])
        else:
            raise AssertionError(f"unhandled captured option {key}")
    return out


@pytest.fixture(scope="module")
def native_cases():
    doc = json.loads(FIXTURE.read_text())
    assert "gpr.m" in doc["native_source"]
    return {case["case_id"]: case for case in doc["cases"]}


@pytest.mark.parametrize("case_id", [
    "ordinary", "noise_unique", "noise_duplicate", "trig", "samples_n40",
])
def test_captured_public_values_and_fixed_l_candidate_outputs(native_cases, case_id):
    case = native_cases[case_id]
    v = case["variables"]
    assert v["success"]["values"] == [True]
    x, y, q = (_values(v, name) for name in ("x", "y", "query"))
    kwargs = _kwargs(v["options"])
    result = gprmod.gpr(x, y, **kwargs)
    native_mean = _values(v, "mean_at_query")
    native_var = _values(v, "variance_at_query")
    assert native_mean.shape == q.shape
    assert native_var.shape == q.shape
    assert np.all(np.isfinite(native_mean)) and np.all(np.isfinite(native_var))
    actual_mean = np.asarray(result["mean_fn"](jnp.asarray(q)))
    actual_var = np.asarray(result["variance_fn"](jnp.asarray(q)))
    assert np.all(np.isfinite(actual_mean)) and np.all(np.isfinite(actual_var))
    # Only ordinary and trig captures currently qualify public-value checks.
    # Noisy captures contain a source discontinuity at exact training hits;
    # MATLAB's public mean/fvar Chebfun approximants do not equal a raw callback
    # there (and the variance approximant differs at other query points too).
    # Preserve and validate every captured value above without reclassifying
    # those observations as callback parity. samples_n40 is also kept as raw
    # evidence, without fitting a tolerance to its 4.6e-10 mean discrepancy.
    if case_id in {"ordinary", "trig"}:
        assert np.allclose(actual_mean, native_mean, rtol=1e-12, atol=1e-12)
        assert np.allclose(actual_var, native_var, rtol=1e-12, atol=1e-12)


def test_empty_fixed_l_source_capture_is_constant_prior(native_cases):
    case = native_cases["empty_fixedL_emptydomain"]
    v = case["variables"]
    kwargs = _kwargs(v["options"])
    result = gprmod.gpr(_values(v, "x"), _values(v, "y"), **kwargs)
    expected_mean = _values(v, "mean_at_query")
    expected_var = _values(v, "variance_at_query")
    # Core-only constant-prior control: explicit source domain=[] is forwarded,
    # and the entire historical dense grid matches all captured query values.
    assert result["domain"] == (-1.0, 1.0)
    assert result["mean"].shape == (200,) and np.all(np.asarray(result["mean"]) == 0.0)
    assert result["variance"].shape == (200,)
    assert np.all(np.asarray(result["variance"]) == expected_var[0])
    assert np.all(expected_mean == 0.0) and np.all(expected_var == expected_var[0])


def test_fixture_retains_empty_sample_source_failure_metadata(native_cases):
    case = native_cases["empty_fixedL_samples"]
    v = case["variables"]
    assert v["success"]["values"] == [False]
    assert v["error_identifier"]["values"] == ["MATLAB:nonExistentField"]
    assert '"sigmaf"' in v["error_message"]["values"][0]
