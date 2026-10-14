"""Diagnostic wrapper comparisons against public native capture fields."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.gpr import gpr_chebfun

FIXTURE = Path(__file__).parent / "data/gpr_fixed_native_hex.json"
if not FIXTURE.exists():
    FIXTURE = Path(__file__).parent / "native/gpr_fixed_native_hex.json"


def _values(variables, key):
    return np.asarray([float.fromhex(v) for v in variables[key]["values"]])


def _options(record):
    pairs = record["options"]["values"]
    out = {}
    i = 0
    while i < len(pairs):
        key = pairs[i]["values"][0].lower()
        i += 1
        if key == "trig":
            out["trig"] = True
            continue
        value = pairs[i]
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
    return out


def test_noisy_native_public_wrapper_comparisons_are_retained():
    doc = json.loads(FIXTURE.read_text())
    cases = {c["case_id"]: c for c in doc["cases"]}
    for case_id in ("noise_unique", "noise_duplicate"):
        case = cases[case_id]
        v = case["variables"]
        x, y, query = (_values(v, k) for k in ("x", "y", "query"))
        mean, variance, _ = gpr_chebfun(x, y, **_options(v))
        actual_mean = np.asarray(mean(jnp.asarray(query))).reshape(-1)
        actual_var = np.asarray(variance(jnp.asarray(query))).reshape(-1)
        native_mean = _values(v, "mean_at_query")
        native_var = _values(v, "variance_at_query")
        assert actual_mean.shape == native_mean.shape
        assert actual_var.shape == native_var.shape
        assert np.all(np.isfinite(actual_mean)) and np.all(np.isfinite(actual_var))
        # These are diagnostic public-Chebfun comparisons. The fixture records
        # source approximation effects at noisy training hits; no tolerance was
        # fitted and this does not assert direct-callback equality.
        print(json.dumps({
            "case": case_id,
            "mean_max_abs_delta": float(np.max(np.abs(actual_mean - native_mean))),
            "variance_max_abs_delta": float(np.max(np.abs(actual_var - native_var))),
            "mean_native": native_mean.tolist(),
            "mean_candidate": actual_mean.tolist(),
            "variance_native": native_var.tolist(),
            "variance_candidate": actual_var.tolist(),
        }, sort_keys=True))
