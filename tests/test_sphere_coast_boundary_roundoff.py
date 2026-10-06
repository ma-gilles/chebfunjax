"""Unrun independent arithmetic-bound and actual source-coast artist tests."""

from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# uses-numpy: renderer fixtures and assertions
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3D  # noqa: E402

from chebfunjax.plotting import (  # noqa: E402
    _draw_sphere_background,
    _sphere_ray_distance2_upper,
    _sphere_visible_vertices,
    plot_earth,
)


@pytest.mark.parametrize("perspective", [False, True])
def test_boundary_roundoff_does_not_hide_front_or_expose_true_interior(perspective):
    matrix = np.array([[0., 1, 0, 0], [0, 0, 1, 0],
                       [0, 0, 0, -1], [-1, 0, 0, 3]]) if perspective else np.array([
                           [0., 1, 0, 0], [0, 0, 1, 0],
                           [-1, 0, 0, 0], [0, 0, 0, 1]])
    one_below = np.nextafter(1., 0.)
    two_below = np.nextafter(one_below, 0.)
    points = np.array([[one_below, 0., 0.], [two_below, 0., 0.],
                       [-one_below, 0., 0.], [.5, 0., 0.],
                       [1.-1e-10, 0., 0.], [1., 0., 0.], [1.1, 0., 0.]])
    np.testing.assert_array_equal(_sphere_visible_vertices(points, matrix, 1.),
                                  [True, True, False, False, False, True, True])


def test_outward_distance_bound_contains_exact_rational_arithmetic():
    # Fraction.from_float treats supplied binary64 data as exact rationals;
    # this oracle does not reuse the interval recurrence or rounded dot sum.
    points = np.array([[.6, .8, 0.], [1., -.7, .3], [1e8, -1e8, .25]])
    rays = np.array([[1., 0., 0.], [-.9, .5, -.2], [-1e8+1., 1e8-1., 0.]])
    parameters = np.array([0., .37, 1.])
    bounds = np.asarray(_sphere_ray_distance2_upper(
        jnp.asarray(points), jnp.asarray(rays), jnp.asarray(parameters)))
    for point, ray, t, bound in zip(points, rays, parameters, bounds):
        exact = sum((Fraction.from_float(float(p)) + Fraction.from_float(float(t))
                     * Fraction.from_float(float(r))) ** 2 for p, r in zip(point, ray))
        assert Fraction.from_float(float(bound)) >= exact


def test_real_source_coast_front_vertices_survive_unit_surface_draw(monkeypatch):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=0, azim=0)
        _draw_sphere_background(ax, scale=1.)
        (line,) = plot_earth(ax, "r--", linewidth=2., label="source data")
        original = np.array(line.get_data_3d(), copy=True)
        finite = np.isfinite(original).all(axis=0)
        # Stay well away from the unqualified horizon/segment approximation.
        # These select separated geometric regions, not pixel thresholds.
        front = finite & (original[0] > .25)
        rear = finite & (original[0] < -.25)
        assert np.any(front & (np.sum(original**2, axis=0) < 1.))
        captured = []
        actual_draw = Line3D.draw

        def observe(artist, renderer):
            if artist is line:
                captured.append(np.array(artist.get_data_3d(), copy=True))
            return actual_draw(artist, renderer)

        monkeypatch.setattr(Line3D, "draw", observe)
        fig.canvas.draw()
        assert np.isfinite(captured[-1][:, front]).all()
        assert np.isnan(captured[-1][:, rear]).all()
        np.testing.assert_array_equal(line.get_data_3d(), original)
        assert line.get_color() == "r" and line.get_linestyle() == "--"
        assert line.get_linewidth() == 2. and line.get_label() == "source data"
    finally:
        plt.close(fig)
