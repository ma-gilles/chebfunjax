"""Independent native R2024b CPU/Painters point-glyph controls.

Native line properties use Chebfun commit7574c77 plot semantics. Captured SVG
geometry qualifies this backend only, including float32 midpoint boundaries.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from chebfunjax.plotting import _source_plot_line_and_points

ROWS = json.loads((Path(__file__).parent / "fixtures/dot_geometry_r2024b.json").read_text())[
    "geometry"
]


@pytest.mark.parametrize("native", ROWS)
def test_independent_native_dot_geometry(native):
    fig, ax = plt.subplots()
    try:
        _source_plot_line_and_points(
            ax,
            [0],
            [0],
            [0],
            [0],
            ".k",
            {"markersize": native["size"], "linewidth": native["line_width"]},
        )
        artist = ax.lines[0]
        marker = artist._marker
        path = marker.get_path().transformed(marker.get_transform())
        radius = path.get_extents().width * artist.get_markersize() / 2
        # Native SVG coordinates are rounded to four decimals (2/3 -> .6667).
        # This bound accounts for export quantization, not scientific tolerance.
        assert abs(radius - native["circle_radius"]) < 5.1e-5
        assert (
            abs(radius + artist.get_markeredgewidth() / 2 - native["outer_extent_radius"]) < 5.1e-5
        )
        assert artist.get_markersize() == native["size"]
    finally:
        plt.close(fig)


def test_source_line_defaults_and_edge_override():
    fig, ax = plt.subplots()
    try:
        _source_plot_line_and_points(
            ax, [0, 1], [0, 1], [0], [0], "b.-", {"linewidth": 1.6, "markersize": 24}
        )
        assert ax.lines[0].get_linewidth() == 1.6
        assert ax.lines[1].get_markersize() == 24
        assert ax.lines[1].get_markeredgewidth() == 1.6
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        _source_plot_line_and_points(ax, [0], [0], [0], [0], ".k", {})
        assert ax.lines[0].get_markersize() == 6
        assert ax.lines[0].get_markeredgewidth() == 0
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        _source_plot_line_and_points(
            ax,
            [0],
            [0],
            [0],
            [0],
            ".k",
            {"markersize": 24, "linewidth": 1.6, "markeredgewidth": 0.7},
        )
        assert ax.lines[0].get_markeredgewidth() == 0.7
    finally:
        plt.close(fig)


def test_zero_size_keeps_invisible_python_marker():
    fig, ax = plt.subplots()
    try:
        _source_plot_line_and_points(ax, [0], [0], [0], [0], ".k", {"markersize": 0})
        assert ax.lines[0].get_markersize() == 0
        assert ax.lines[0].get_markeredgewidth() == 0
        fig.canvas.draw()
    finally:
        plt.close(fig)
