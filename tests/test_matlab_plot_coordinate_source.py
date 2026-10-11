"""Native R2025b builtin plot coordinate controls, supplemental to Chebfun tests.

The literal Lorenz source plots complex numerator roots against scalar zero.
No ODE or rational approximation is evaluated by these controls.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.plotting import matlab_plot

_CASES = json.loads(
    (Path(__file__).parent / "references/lorenz_plot_coordinates_r2025b_20261011.json").read_text()
)["cases"]
_X = np.array([1, 2, 3])
_Y = np.array([4, 5, 6])
_CX = np.array([1 + 2j, 2 - 3j, 3 + 4j])
_CY = np.array([4 + 2j, 5 - 3j, 6 + 4j])
_INPUTS = {
    "real_scalarY": (_X, 0),
    "complexX_scalarY": (_CX, 0),
    "complexX_vectorY": (_CX, _Y),
    "complexY": (_X, _CY),
    "bothcomplex": (_CX, _CY),
    "zeroimagX": (_X.astype(complex), _Y),
    "zeroimagY": (_X, _Y.astype(complex)),
}


@pytest.mark.parametrize("record", _CASES[:7], ids=[r["name"] for r in _CASES[:7]])
def test_native_coordinate_warning_and_artist_data(record):
    x, y = _INPUTS[record["name"]]
    xcopy = np.array(x, copy=True)
    ycopy = np.array(y, copy=True)
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            matlab_plot(x, y, "ok", ax=ax)
        expected = record["warning_message"]
        assert [str(w.message) for w in seen] == ([expected] if expected else [])
        assert len(ax.lines) == len(record["lines"])
        for line, oracle in zip(ax.lines, record["lines"]):
            np.testing.assert_array_equal(line.get_xdata(), np.atleast_1d(oracle["x"]))
            np.testing.assert_array_equal(line.get_ydata(), np.atleast_1d(oracle["y"]))
        np.testing.assert_array_equal(x, xcopy)
        np.testing.assert_array_equal(y, ycopy)
    finally:
        plt.close(fig)


def test_scalar_x_symmetric_adapter_and_warning_filters_unchanged():
    """Supplemental symmetric scalar adapter; native packet directly covers scalar Y."""
    before = list(warnings.filters)
    fig, ax = plt.subplots()
    try:
        matlab_plot(0, np.array([1, 2, 3]), "ok", ax=ax)
        assert len(ax.lines) == 3
        for i, line in enumerate(ax.lines):
            np.testing.assert_array_equal(line.get_xdata(), [0])
            np.testing.assert_array_equal(line.get_ydata(), [i + 1])
        assert warnings.filters == before
    finally:
        plt.close(fig)


def test_unrelated_input_adapter_warning_survives():
    class WarnedInput:
        def __array__(self, dtype=None, copy=None):
            warnings.warn("unrelated input adapter warning", UserWarning)
            return np.array([1 + 2j, 2 - 3j, 3 + 4j], dtype=dtype)

    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            matlab_plot(WarnedInput(), [4, 5, 6], "ok", ax=ax)
        assert [str(w.message) for w in seen] == [
            "unrelated input adapter warning",
            "Imaginary parts of complex X and/or Y arguments ignored.",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("scalar_is_x", [True, False])
def test_full_matrix_scalar_pair_retains_existing_fallback(monkeypatch, scalar_is_x):
    """Structural control: unqualified full matrices must not be flattened."""
    import chebfunjax.plotting as plotting
    matrix = np.arange(6).reshape(2, 3)
    captured = []
    monkeypatch.setattr(plotting, "_plot_curve", lambda ax, x, y, fmt, kw: captured.append((x, y)))
    fig, ax = plt.subplots()
    try:
        x, y = (0, matrix) if scalar_is_x else (matrix, 0)
        matlab_plot(x, y, "ok", ax=ax)
        assert len(captured) == 1
        np.testing.assert_array_equal(captured[0][0], x)
        np.testing.assert_array_equal(captured[0][1], y)
    finally:
        plt.close(fig)
