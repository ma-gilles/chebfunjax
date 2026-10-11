"""Native matrix-logarithm warning controls; no original assertion count claimed."""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import logm

_REF = Path(__file__).parent / "references"
_META = json.loads((_REF / "floquet_logm_native_r2025b_20261011.json").read_text())


@pytest.mark.parametrize("index", range(7), ids=[x["name"] for x in _META["cases"]])
def test_native_logm_warning_and_unchanged_provider_result(index, tmp_path):
    fixture = np.load(_REF / "floquet_logm_native_r2025b_20261011.npz")
    script = Path(__file__).parents[1] / "examples/ode-linear/floquet.py"
    spec = importlib.util.spec_from_file_location("floquet_logm_source", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = fixture[f"case{index}_A"]
    before = matrix.copy()
    with warnings.catch_warnings(record=True) as legacy:
        warnings.simplefilter("always")
        original = logm(matrix)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        actual = module._source_logm(matrix)
    native_message = _META["cases"][index]["warning_message"]
    assert [str(w.message) for w in observed] == ([native_message] if native_message else []) + [
        str(w.message) for w in legacy
    ]
    assert actual.dtype == original.dtype and actual.shape == original.shape
    assert actual.tobytes() == original.tobytes()
    np.testing.assert_array_equal(matrix, before)
    if index == 0:
        assert (actual / module.T).tobytes() == (original / module.T).tobytes()
        output = tmp_path
        (output / "ACTUAL_WARNING.json").write_text(
            json.dumps(
                {
                    "message": str(observed[0].message),
                    "category": observed[0].category.__name__,
                    "condition_input": "actual fresh source PhiT",
                    "scope": "Captured warning event on saved actualsourceinput; not a full source replay",
                },
                indent=2,
            )
            + "\n"
        )
        (output / "warning.stdout").write_text("Warning: " + str(observed[0].message) + "\n")
