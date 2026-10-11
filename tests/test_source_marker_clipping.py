"""Document-backed marker policy, not historical MATLAB pixel parity."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

from chebfunjax.plotting import _source_marker_artist, _source_plot_line_and_points


def canvas():
    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = fig.add_axes([0.25, 0.25, 0.5, 0.5])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def red(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return (rgba[:, :, 0] > 180) & (rgba[:, :, 1] < 100)


def test_edge_corner_outside_pixels_and_original_data():
    fig, ax = canvas()
    x = np.array([0.0, 0.0, -0.01, 1.0, np.nan])
    y = np.array([0.5, 0.0, 0.8, 1.0, 0.3])
    artist = _source_marker_artist(
        ax, x, y, dict(marker="o", markersize=14, color="red", linestyle="None", markeredgewidth=0)
    )
    original_x, original_y = artist.get_data()
    pixels = red(fig)
    assert pixels[100, 45]  # complete left-edge glyph
    assert pixels[155, 45]  # complete bottom-left corner
    assert pixels[45, 155]  # complete top-right corner
    assert not pixels[70, 45]  # outside center must not leak a partial glyph
    assert artist.get_data()[0] is original_x and artist.get_data()[1] is original_y
    np.testing.assert_array_equal(artist.get_xdata(), x)
    np.testing.assert_array_equal(artist.get_ydata(), y)
    assert artist.get_clip_on()
    plt.close(fig)


@pytest.mark.parametrize("mode", ["zoom", "reversed", "log", "reversed_log"])
def test_limits_are_recomputed(mode):
    fig, ax = canvas()
    x = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    y = np.ones(5) * 0.5
    artist = _source_marker_artist(
        ax, x, y, dict(marker="o", markersize=14, color="red", linestyle="None", markeredgewidth=0)
    )
    red(fig)
    if "log" in mode:
        ax.set_xscale("log")
        ax.set_xlim(0.1, 10.0)
        expected = np.array([False, True, True, True, False])
    else:
        ax.set_xlim(0.1, 1.0)
        expected = np.array([False, True, True, False, False])
    if "reversed" in mode:
        ax.invert_xaxis()
        ax.invert_yaxis()
    actual = red(fig)
    artist.set_visible(False)
    ax.plot(
        np.where(expected, x, np.nan),
        np.where(expected, y, np.nan),
        "ro",
        markersize=14,
        markeredgewidth=0,
        clip_on=False,
    )
    oracle = red(fig)
    np.testing.assert_array_equal(actual, oracle)
    np.testing.assert_array_equal(artist.get_xdata(), x)
    plt.close(fig)


@pytest.mark.parametrize("kind", ["off", "path", "box"])
def test_explicit_clip_policy_is_preserved(kind):
    fig, ax = canvas()
    artist = _source_marker_artist(
        ax, [-0.03, 0.0], [0.5, 0.8], dict(marker="o", markersize=14, color="red", linestyle="None")
    )
    if kind == "off":
        artist.set_clip_on(False)
    if kind == "path":
        artist.set_clip_path(Circle((0.5, 0.5), 0.1, transform=ax.transData))
    if kind == "box":
        artist.set_clip_box(Bbox.from_bounds(60, 60, 60, 60))
    actual = red(fig)
    artist.set_visible(False)
    (oracle,) = ax.plot([-0.03, 0.0], [0.5, 0.8], "ro", markersize=14)
    if kind == "off":
        oracle.set_clip_on(False)
    if kind == "path":
        oracle.set_clip_path(Circle((0.5, 0.5), 0.1, transform=ax.transData))
    if kind == "box":
        oracle.set_clip_box(Bbox.from_bounds(60, 60, 60, 60))
    np.testing.assert_array_equal(actual, red(fig))
    plt.close(fig)


def test_mixed_line_markers_metadata_cycle_and_legend():
    fig, ax = canvas()
    _source_plot_line_and_points(
        ax,
        [-1, 2],
        [0.5, 0.5],
        [0, 1],
        [0.5, 0.5],
        "-o",
        dict(color="red", label="source", markersize=14),
    )
    line, points = ax.lines
    assert line.get_clip_on() and line.get_clip_box() is not None
    assert line.get_color() == points.get_color() == "red"
    assert line.get_label() == "source" and points.get_label() == "_nolegend_"
    assert ax.get_legend_handles_labels() == ([line], ["source"])
    assert not red(fig)[100, 30]  # ordinary line remains clipped
    plt.close(fig)


def test_actual_legend_repeated_draw_does_not_notify_original_callbacks():
    fig, ax = canvas()
    artist = _source_marker_artist(
        ax,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        dict(marker="o", color="red", linestyle="None", label="points"),
    )
    data = artist.get_data()
    callbacks = artist._callbacks
    callback_events, stale_events = [], []
    artist.add_callback(lambda changed: callback_events.append(changed))
    original_stale_callback = artist.stale_callback

    def observe_stale(changed, value):
        stale_events.append((changed, value))
        original_stale_callback(changed, value)

    artist.stale_callback = observe_stale
    legend = ax.legend()
    callback_events.clear()
    stale_events.clear()
    for _ in range(3):
        fig.canvas.draw()
        assert artist.get_data()[0] is data[0] and artist.get_data()[1] is data[1]
        assert artist._callbacks is callbacks
        assert artist.stale_callback is observe_stale
        assert artist.get_clip_on()
    assert not callback_events and not stale_events
    assert [type(handle).__name__ for handle in legend.get_lines()] == ["Line2D"]
    np.testing.assert_array_equal(artist.get_xdata(), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(artist.get_ydata(), [0.0, 0.5, 1.0])
    plt.close(fig)


def test_visible_draw_clears_stale_and_hidden_draw_preserves_it():
    fig, ax = plt.subplots()
    source = _source_marker_artist(ax, [0.0, 1.0], [0.0, 1.0], dict(marker="o", linestyle="None"))
    (ordinary,) = ax.plot([0.0, 1.0], [0.0, 1.0], "o")
    fig.canvas.draw()
    assert not source.stale and not ordinary.stale
    source.set_visible(False)
    ordinary.set_visible(False)
    assert source.stale and ordinary.stale
    fig.canvas.draw()
    assert source.stale and ordinary.stale
    plt.close(fig)
