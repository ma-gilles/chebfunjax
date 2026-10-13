"""Plain-array MATLAB plot dot routing, grounded in native glyph controls."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.plotting import matlab_plot

_GEOMETRY = json.loads(
    (Path(__file__).parent / "fixtures/dot_geometry_r2024b.json").read_text()
)["geometry"]


def _native_radius(artist):
    marker = artist._marker
    path = marker.get_path().transformed(marker.get_transform())
    return path.get_extents().width * artist.get_markersize() / 2


def test_plain_array_dot_preserves_native_glyph_coordinates_and_color():
    native = next(row for row in _GEOMETRY if row["size"] == 12)
    x = np.array([0.25, 1.5, 3.0])
    y = np.array([-2.0, 4.0, 0.5])
    fig, ax = plt.subplots()
    try:
        matlab_plot(x, y, ".r", ax=ax, markersize=12)
        assert len(ax.lines) == 1
        point = ax.lines[0]
        np.testing.assert_array_equal(point.get_xdata(), x)
        np.testing.assert_array_equal(point.get_ydata(), y)
        assert point.get_color() == "r"
        assert point.get_linestyle() in ("None", "none", "")
        assert point.get_markersize() == 12
        assert abs(_native_radius(point) - native["circle_radius"]) < 5.1e-5
    finally:
        plt.close(fig)


def test_plain_array_dot_dash_retains_line_and_marker_semantics():
    x = np.array([1, 2, 3])
    y = np.array([2, 1, 4])
    fig, ax = plt.subplots()
    try:
        matlab_plot(x, y, ".-", ax=ax, markersize=10, linewidth=0.5)
        assert len(ax.lines) == 2
        line, points = ax.lines
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        np.testing.assert_array_equal(points.get_xdata(), x)
        np.testing.assert_array_equal(points.get_ydata(), y)
        assert line.get_linestyle() == "-"
        assert line.get_marker() in (None, "None", "")
        assert points.get_linestyle() in ("None", "none", "")
        assert points.get_markersize() == 10
        assert points.get_color() == line.get_color()
        assert line.get_linewidth() == 0.5
    finally:
        plt.close(fig)


def test_plain_dot_uses_one_default_color_cycle_step_and_default_line():
    x = np.array([1, 2, 3])
    y = np.array([2, 1, 4])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()
    try:
        matlab_plot(x, y, ax=ax, marker=".", markersize=8)
        matlab_plot(x, y, ax=ax, marker=".", markersize=8)
        assert len(ax.lines) == 4
        for offset, expected_color in ((0, colors[0]), (2, colors[1])):
            line, points = ax.lines[offset:offset + 2]
            assert line.get_linestyle() == "-"
            assert line.get_marker() in (None, "None", "")
            assert line.get_color() == expected_color
            assert points.get_color() == expected_color
            assert points.get_markersize() == 8
    finally:
        plt.close(fig)


def test_plain_matrix_dots_follow_native_column_layout():
    x = np.array([[0, 10], [1, 11], [2, 12]])
    y = np.array([[2, 4], [3, 5], [4, 6]])
    fig, ax = plt.subplots()
    try:
        matlab_plot(x, y, ".k", ax=ax, markersize=8)
        assert len(ax.lines) == 2
        for j, artist in enumerate(ax.lines):
            np.testing.assert_array_equal(artist.get_xdata(), x[:, j])
            np.testing.assert_array_equal(artist.get_ydata(), y[:, j])
            assert artist.get_color() == "k"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt,size", [("o", 12), (".", 24)])
def test_unqualified_plain_markers_keep_matplotlib_artist(fmt, size):
    x = np.array([1, 2, 3])
    y = np.array([2, 1, 4])
    fig, ax = plt.subplots()
    try:
        matlab_plot(x, y, fmt, ax=ax, markersize=size)
        assert len(ax.lines) == 1
        artist = ax.lines[0]
        np.testing.assert_array_equal(artist.get_xdata(), x)
        np.testing.assert_array_equal(artist.get_ydata(), y)
        if fmt == "o":
            assert artist.get_marker() == "o"
        else:
            assert artist.get_marker() == "."
        assert artist.get_markersize() == size
    finally:
        plt.close(fig)
