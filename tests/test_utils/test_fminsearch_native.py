"""Fresh native R2025b traces for Chebfun extrema and Needle tolerance policies."""
import json
import os
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.utils._fminsearch import fminsearch

FIXTURE = json.loads(Path(__file__).with_name("fminsearch_native_r2025b.json").read_text())


@pytest.mark.parametrize("caseid", range(5))
def test_native_callback_trace_and_stopping(caseid):
    expected = FIXTURE["cases"][caseid]
    seen, values = [], []

    def objective(x):
        if caseid in (2, 4):
            value = jnp.asarray(0., dtype=jnp.float64)
        elif caseid == 0:
            value = (x[0]-.25)**2+(x[1]+.5)**2
        else:
            value = (x[0]-.5)**2+(x[1]+.25)**2
            if caseid == 3:
                value = 1e12*value
        seen.append(x)
        values.append(value)
        return value

    result = fminsearch(objective, expected["initial"],
                        tol_x=expected["tol_x"], tol_fun=expected["tol_fun"],
                        max_evaluations=int(expected["max_evaluations"]))
    calls = jnp.stack(seen)
    fvalues = jnp.stack(values)
    if directory := os.environ.get("NEEDLE_FMINSEARCH_TRACE_DIR"):
        Path(directory, f"case{caseid}.json").write_text(json.dumps({
            "calls": calls.tolist(), "values": fvalues.tolist(),
            "x": result.x.tolist(), "fun": float(result.fun),
            "exitflag": result.exitflag, "iterations": result.iterations,
            "evaluations": result.evaluations}, indent=2)+"\n")
    assert (result.iterations, result.evaluations, result.exitflag) == (
        expected["iterations"], expected["evaluations"], expected["exitflag"])
    # Exact binary64 callbacks include signed zeros and every simplex branch.
    assert calls.shape == jnp.asarray(expected["calls"]).shape
    assert bool(jnp.array_equal(calls.view(jnp.uint64),
                               jnp.asarray(expected["calls"], dtype=jnp.float64).view(jnp.uint64)))
    assert bool(jnp.array_equal(fvalues.view(jnp.uint64),
                               jnp.asarray(expected["values"], dtype=jnp.float64).view(jnp.uint64)))
    assert bool(jnp.array_equal(result.x.view(jnp.uint64),
                               jnp.asarray(expected["x"], dtype=jnp.float64).view(jnp.uint64)))
    assert result.fun == expected["fun"]


@pytest.mark.parametrize("name,value", [("tol_x", 0.), ("tol_fun", float("nan")),
                                        ("tol_x", float("inf")), ("tol_fun", [1e-4])])
def test_unsupported_tolerances_fail_before_objective(name, value):
    def objective(x):
        raise AssertionError("unsupported tolerance reached objective")
    with pytest.raises(ValueError, match="finite scalar"):
        fminsearch(objective, [0., 0.], **{name: value})
