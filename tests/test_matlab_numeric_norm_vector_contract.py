"""Small builtin-native discriminator for the MEAN test norm adapter."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def test_vector_and_matrix_norm_adapter():
    root = Path(__file__).parent
    fixture = json.loads((root / "fixtures/builtin_numeric_norm_vector.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "mean_norm_adapter", root / "test_matlab_port/chebfun/test_mean_matlab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for values, expected in [
        (np.array([[1, 2, 3]]), fixture["row_inf"]),
        (np.array([[1], [2], [3]]), fixture["column_inf"]),
        (np.array([1, 2, 3]), fixture["row_inf"]),
        (np.array([[1, 2, 3], [4, 5, 6]]), fixture["matrix_inf"]),
        (np.array([[1 + 1j, 2 - 2j, 3]]), fixture["complex_row_inf"]),
    ]:
        assert module.matrix_inf(values) == expected
