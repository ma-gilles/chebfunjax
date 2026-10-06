"""Geometric and real-artist controls for sphere camera/coast rendering."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# uses-numpy: Matplotlib artist inspection and independent assertions.
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3D  # noqa: E402

from chebfunjax.plotting import (  # noqa: E402
    _draw_sphere_background,
    _sphere_visible_vertices,
    matlab_view,
    plot_earth,
)


@pytest.mark.parametrize("azimuth,front", [(0., [0., -1., 0.]),
                                          (90., [1., 0., 0.]),
                                          (180., [0., 1., 0.])])
@pytest.mark.parametrize("projection", ["ortho", "persp"])
def test_matlab_cardinal_views_show_expected_geographic_hemisphere(azimuth, front, projection):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type(projection)
        matlab_view(ax, azimuth, 0)
        points = np.asarray([front, -np.asarray(front), [np.nan]*3])
        np.testing.assert_array_equal(
            _sphere_visible_vertices(points, ax.get_proj(), 1.), [True, False, False])
    finally:
        plt.close(fig)


def test_source_radius_can_expose_coast_behind_tangent_plane():
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=0, azim=0)
        points = np.asarray([[-.1, 1.001, 0.], [-.1, .995, 0.]])
        # Both points have negative x. The first ray misses a unit sphere;
        # the second misses the source's0.99 contour sphere but hits unit1.
        np.testing.assert_array_equal(
            _sphere_visible_vertices(points, ax.get_proj(), 1.), [True, False])
        np.testing.assert_array_equal(
            _sphere_visible_vertices(points, ax.get_proj(), .99), [True, True])
    finally:
        plt.close(fig)


def test_real_coast_artist_draw_clips_updates_and_preserves_source_data(monkeypatch):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_proj_type("ortho")
        _draw_sphere_background(ax, scale=.99)
        (line,) = plot_earth(ax, "k-", linewidth=2)
        original = np.array(line.get_data_3d(), copy=True)
        captured = []
        original_draw = Line3D.draw

        def observe_draw(artist, renderer):
            if artist is line:
                captured.append(np.array(artist.get_data_3d(), copy=True))
            return original_draw(artist, renderer)

        monkeypatch.setattr(Line3D, "draw", observe_draw)
        matlab_view(ax, 90., 0.)
        fig.canvas.draw()
        first = captured[-1]
        assert np.isfinite(first).all(axis=0).any()
        assert np.isnan(first).any(axis=0).sum() > np.isnan(original).any(axis=0).sum()
        assert line.get_zorder() > max(c.get_zorder() for c in ax.collections)
        np.testing.assert_array_equal(line.get_data_3d(), original)
        assert line.get_linewidth() == 2
        matlab_view(ax, -90., 0.)
        fig.canvas.draw()
        assert not np.array_equal(np.isfinite(first), np.isfinite(captured[-1]))
        np.testing.assert_array_equal(line.get_data_3d(), original)
    finally:
        plt.close(fig)


def test_visibility_interior_boundary_exterior_and_camera_point():
    # Exact synthetic Matplotlib-form matrices; eye is +infinity or (3,0,0).
    ortho = np.array([[0., 1, 0, 0], [0, 0, 1, 0],
                      [-1, 0, 0, 0], [0, 0, 0, 1]])
    perspective = np.array([[0., 1, 0, 0], [0, 0, 1, 0],
                            [0, 0, 0, -1], [-1, 0, 0, 3]])
    points = np.array([[2., 0, 0], [.5, 0, 0], [1., 0, 0],
                       [-1., 0, 0], [0, 2., 0], [3., 0, 0]])
    np.testing.assert_array_equal(
        _sphere_visible_vertices(points, ortho, 1.),
        [True, False, True, False, True, True])
    np.testing.assert_array_equal(
        _sphere_visible_vertices(points, perspective, 1.),
        [True, False, True, False, True, False])


@pytest.mark.parametrize("projection", ["ortho", "persp"])
def test_matrix_visibility_includes_roll_and_anisotropic_world_scaling(projection):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-2, 2), ylim=(-1, 1), zlim=(-3, 3))
        ax.set_proj_type(projection)
        ax.view_init(elev=0, azim=0, roll=37)
        np.testing.assert_array_equal(_sphere_visible_vertices(
            [[1., 0, 0], [-1., 0, 0]], ax.get_proj(), 1.), [True, False])
    finally:
        plt.close(fig)


def test_matlab_view_preserves_projection_zoom_and_axes_layout():
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("persp", focal_length=2.5)
        ax.set_box_aspect([1., 2., 3.], zoom=.7)
        aspect = ax._box_aspect.copy()
        position = ax.get_position().bounds
        matlab_view(ax, 50, 5)
        assert ax.azim == -40 and ax.elev == 5
        assert ax._focal_length == 2.5
        np.testing.assert_array_equal(ax._box_aspect, aspect)
        assert ax.get_position().bounds == position
    finally:
        plt.close(fig)


def test_artist_lifecycle_no_phantom_occluder_and_user_zorder(monkeypatch):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        (line,) = plot_earth(ax, "r--", linewidth=2, label="source coast")
        original = np.array(line.get_data_3d(), copy=True)
        captured = []
        real_draw = Line3D.draw

        def observe(artist, renderer):
            if artist is line:
                captured.append(np.array(artist.get_data_3d(), copy=True))
            return real_draw(artist, renderer)

        monkeypatch.setattr(Line3D, "draw", observe)
        # Standalone and unrelated geometry must not create a sphere mask.
        fig.canvas.draw()
        np.testing.assert_array_equal(captured[-1], original)
        _draw_sphere_background(ax)
        sphere = ax.collections[-1]
        fig.canvas.draw()
        assert not np.array_equal(np.isfinite(captured[-1]), np.isfinite(original))
        for mode in ("transparent", "hidden", "removed"):
            sphere.set_visible(True)
            sphere.set_alpha(1.)
            if mode == "transparent":
                sphere.set_alpha(.5)
            elif mode == "hidden":
                sphere.set_visible(False)
            else:
                sphere.remove()
            fig.canvas.draw()
            np.testing.assert_array_equal(captured[-1], original)
        _draw_sphere_background(ax)
        line.set_zorder(0)
        fig.canvas.draw()
        assert line.get_zorder() == 0
        assert line.get_label() == "source coast"
        assert line.get_color() == "r"
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 2
        np.testing.assert_array_equal(line.get_data_3d(), original)
    finally:
        plt.close(fig)


def test_original_line_data_restored_if_renderer_raises(monkeypatch):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        _draw_sphere_background(ax)
        (line,) = plot_earth(ax)
        original = np.array(line.get_data_3d(), copy=True)
        real_draw = Line3D.draw

        def fail(artist, renderer):
            if artist is line:
                raise RuntimeError("renderer failure")
            return real_draw(artist, renderer)

        monkeypatch.setattr(Line3D, "draw", fail)
        with pytest.raises(RuntimeError, match="renderer failure"):
            fig.canvas.draw()
        np.testing.assert_array_equal(line.get_data_3d(), original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("manual_order", [True, False])
def test_explicit_axes_order_and_unrelated_collection_are_respected(manual_order):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d", computed_zorder=not manual_order)
        _draw_sphere_background(ax)
        sphere = ax.collections[-1]
        sphere.set_zorder(5)
        unrelated = ax.scatter([0.], [0.], [2.], zorder=1000)
        (line,) = plot_earth(ax, "k-")
        if manual_order:
            assert line.get_zorder() == 2
        else:
            assert sphere.get_zorder() < line.get_zorder() < unrelated.get_zorder()
        line.set_zorder(0)
        assert line.get_zorder() == 0
    finally:
        plt.close(fig)


def test_actual_plot_sphere_surface_registers_coast_occluder():
    from chebfunjax.plotting import plot_sphere

    class AnalyticField:
        def __call__(self, longitude, colatitude):
            return np.cos(np.asarray(colatitude)) + .25*np.sin(np.asarray(colatitude))*np.cos(np.asarray(longitude))

    fig, ax = plot_sphere(AnalyticField(), n_pts=12)
    try:
        surface = ax.collections[-1]
        (line,) = plot_earth(ax)
        original = np.array(line.get_data_3d(), copy=True)
        fig.canvas.draw()
        assert line.get_zorder() > surface.get_zorder()
        np.testing.assert_array_equal(line.get_data_3d(), original)
        surface.set_alpha(.5)
        assert line.get_zorder() == 2
    finally:
        plt.close(fig)
