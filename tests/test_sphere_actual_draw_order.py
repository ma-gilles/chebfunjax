"""Real painter-order and raster controls for native sphere line geometry."""
# uses-numpy: independent graphics fixtures, pixel buffers and assertions.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

from chebfunjax.plotting import (
    _draw_sphere_background,
    contour_sphere,
    matlab_view,
    plot_earth,
)


def observe_draws(monkeypatch, ax):
    events, vertices = [], []
    line_draw, surface_draw = Line3D.draw, Poly3DCollection.draw

    def line(artist, renderer):
        if artist in ax.lines:
            events.append(("line", artist))
            vertices.append(np.array(artist.get_data_3d(), copy=True))
        return line_draw(artist, renderer)

    def surface(artist, renderer):
        if artist in ax.collections:
            events.append(("surface", artist))
        return surface_draw(artist, renderer)

    monkeypatch.setattr(Line3D, "draw", line)
    monkeypatch.setattr(Poly3DCollection, "draw", surface)
    return events, vertices


def red_pixels(fig):
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return (rgba[..., 0] > 150) & (rgba[..., 1] < 100) & (rgba[..., 2] < 100)


@pytest.mark.parametrize("elevation", [0., 5.])
def test_source_coast_actual_painter_order_and_pixels(monkeypatch, elevation):
    fig = plt.figure(figsize=(6, 2.7))
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_axis_off()
        _draw_sphere_background(ax, color="white", scale=1.)
        (line,) = plot_earth(ax, "r-", linewidth=2.)
        original = np.array(line.get_data_3d(), copy=True)
        matlab_view(ax, 50., elevation)
        events, vertices = observe_draws(monkeypatch, ax)
        fig.canvas.draw()
        assert events.index(("surface", ax.collections[0])) < events.index(("line", line))
        assert line.zorder == line.get_zorder() > ax.collections[0].zorder
        assert np.isfinite(vertices[-1]).any() and np.isnan(vertices[-1]).any()
        np.testing.assert_array_equal(line.get_data_3d(), original)
        assert red_pixels(fig).any()
        line.set_visible(False)
        fig.canvas.draw()
        assert not red_pixels(fig).any()
    finally:
        plt.close(fig)


class LatitudeField:
    def sample(self, m, n):
        return np.broadcast_to(np.cos(np.linspace(0., np.pi, n))[:, None], (n, m))


def test_contour_front_back_geometry_draw_order_and_pixels(monkeypatch):
    fig, ax = contour_sphere(LatitudeField(), n_pts=33, levels=[0.], fmt="r-", linewidth=2.)
    try:
        ax.set_axis_off()
        ax.set_proj_type("ortho")
        matlab_view(ax, 90., 0.)  # eye at positive x; equator has known front/back.
        originals = [np.array(line.get_data_3d(), copy=True) for line in ax.lines]
        events, vertices = observe_draws(monkeypatch, ax)
        fig.canvas.draw()
        assert len(vertices) == len(originals) > 0
        for line, original, displayed in zip(ax.lines, originals, vertices):
            assert events.index(("surface", ax.collections[0])) < events.index(("line", line))
            assert np.isfinite(displayed[:, original[0] > .25]).all()
            assert np.isnan(displayed[:, original[0] < -.25]).all()
            np.testing.assert_array_equal(line.get_data_3d(), original)
            np.testing.assert_allclose(np.sum(original**2, axis=0), 1., rtol=0., atol=8*np.finfo(float).eps)
            assert line.get_linewidth() == 2.
        assert red_pixels(fig).any()
    finally:
        plt.close(fig)


def test_explicit_order_keeps_actual_user_painter_choice(monkeypatch):
    fig, ax = contour_sphere(LatitudeField(), n_pts=17, levels=[0.], fmt="r-", zorder=0.)
    try:
        ax.set_axis_off()
        events, _ = observe_draws(monkeypatch, ax)
        fig.canvas.draw()
        for line in ax.lines:
            assert line.zorder == line.get_zorder() == 0.
            assert events.index(("line", line)) < events.index(("surface", ax.collections[0]))
            line.set_zorder(7.)
            assert line.zorder == line.get_zorder() == 7.
    finally:
        plt.close(fig)


def test_none_and_direct_numeric_zorder_draw_safely(monkeypatch):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        _draw_sphere_background(ax, color="white", scale=1.)
        (line,) = plot_earth(ax, "r-")
        events, _ = observe_draws(monkeypatch, ax)
        line.set_zorder(None)
        assert line.zorder == line.get_zorder() == Line3D.zorder
        fig.canvas.draw()
        assert events.index(("line", line)) < events.index(("surface", ax.collections[0]))
        events.clear()
        line.zorder = 7.
        assert line.zorder == line.get_zorder() == 7.
        fig.canvas.draw()
        assert events.index(("surface", ax.collections[0])) < events.index(("line", line))
    finally:
        plt.close(fig)
