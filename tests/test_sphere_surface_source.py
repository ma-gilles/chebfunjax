"""Source sphere artists and actual website presentation controls."""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# uses-numpy: renderer fixtures and exact artist/source assertions.
from chebfunjax._sphere_surface import InterpolatedSphereSurface  # noqa: E402
from chebfunjax.plotting import (  # noqa: E402
    _matlab_facecolors,
    _normalize_values,
    contour_sphere,
    plot_sphere,
    save_chebfun_figure,
)
from chebfunjax.spherefun import Spherefun  # noqa: E402

PAGE_PATH = Path(__file__).resolve().parents[1] / "examples/sphere/sphereheatconduction.py"
spec = importlib.util.spec_from_file_location("sphere_renderer_page", PAGE_PATH)
page = importlib.util.module_from_spec(spec)
sys.modules["sphere_renderer_page"] = page
spec.loader.exec_module(page)


class ColorField:
    def __call__(self, longitude, theta):
        return np.sin(np.asarray(theta)) * np.cos(np.asarray(longitude))

    def sample(self, longitude_count, theta_count):
        l = np.linspace(-np.pi, np.pi, longitude_count, endpoint=False)
        t = np.linspace(0., np.pi, theta_count)
        ll, tt = np.meshgrid(l, t)
        return self(ll, tt)


def test_source_vertex_colors_and_grid_are_retained():
    x, y = np.meshgrid(np.array([-1., 0., 1.]), np.array([-1., 1.]))
    z = x * 0
    colors = np.arange(24, dtype=float).reshape(2, 3, 4) / 24
    artist = InterpolatedSphereSurface(x, y, z, colors)
    np.testing.assert_array_equal(artist._source_vertices, np.stack((x, y, z), axis=-1))
    np.testing.assert_array_equal(artist._source_rgba, colors)
    np.testing.assert_array_equal(artist._triangle_indices,
                                  [[0, 1, 4], [1, 2, 5], [0, 4, 3], [1, 5, 4]])
    np.testing.assert_array_equal(artist._triangle_rgba,
                                  colors.reshape(-1, 4)[artist._triangle_indices])


def test_default_source_camera_and_existing_axes():
    fig, ax = plot_sphere(ColorField())
    assert (ax.elev, ax.azim) == (30., -127.5)
    assert np.isinf(ax._focal_length)
    assert ax.collections[0]._source_vertices.shape == (200, 200, 3)
    plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=13., azim=41.)
    plot_sphere(ColorField(), ax=ax)
    assert (ax.elev, ax.azim) == (13., 41.)
    plt.close(fig)


def test_gouraud_sphere_draw_and_camera_updates(tmp_path):
    field = ColorField()
    fig, ax, mappable = plot_sphere(field, title='Time 0.25', return_mappable=True,
                                   cmap=page._website_colormap(), clim=(-1., 1.5))
    artist = ax.collections[0]
    fig.canvas.draw()
    first = artist._projected_triangles.copy()
    ax.view_init(elev=0., azim=0.)
    fig.canvas.draw()
    assert np.isfinite(artist._projected_triangles).all()
    assert not np.array_equal(first, artist._projected_triangles)
    assert artist._triangle_xyz.shape == (2*199*199, 3, 3)
    contour_sphere(field, ax=ax, hold=True, levels=[.1, .1], fmt='b-')
    assert ax.lines and all(line.get_color() == 'b' for line in ax.lines)
    page._website_layout(fig, ax, mappable, 'Time 0.25', True)
    ax.set_axis_off()
    save_chebfun_figure(fig, tmp_path/'analytic_gouraud.png', size=(610, 276))
    fig.set_dpi(100.)
    fig.canvas.draw()
    box = ax.title.get_window_extent(fig.canvas.get_renderer())
    assert box.y0 >= 0 and box.y1 <= 276.00001
    np.testing.assert_allclose(fig.axes[1].get_position().bounds,
                               (486/610, 31/276, 40/610, 220/276), rtol=0, atol=1e-15)
    assert [text.get_text() for text in fig.axes[1].get_yticklabels()] == ['−1', '−0.5', '0', '0.5', '1', '1.5']
    with Image.open(tmp_path/'analytic_gouraud.png') as im:
        assert im.size == (610, 276)
    plt.close(fig)


def test_source_harmonic_initial_render_only(tmp_path):
    # Actual original initial field, no time steps or reference-image reuse.
    initial = Spherefun.sphharm(6, 0) + np.sqrt(14/11) * Spherefun.sphharm(6, 5)
    fig, ax, mappable = initial.plot(return_mappable=True, clim=(-1., 1.5),
                                     cmap=page._website_colormap())
    page._website_layout(fig, ax, mappable, '', False)
    ax.set_axis_off()
    save_chebfun_figure(fig, tmp_path/'harmonic_initial_renderer.png', size=(610, 276))
    fig.canvas.draw()
    assert [text.get_text() for text in fig.axes[1].get_yticklabels()] == ['−1', '−0.5', '0', '0.5', '1', '1.5']
    with Image.open(tmp_path/'harmonic_initial_renderer.png') as im:
        rgb = np.asarray(im.convert('RGB'))
    mask = np.ptp(rgb[:, :427].astype(int), axis=-1) > 16
    yy, xx = np.nonzero(mask)
    observed_bounds = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
    # Website raster geometry target, independent of numerical accuracy.
    assert observed_bounds == [171, 61, 316, 206]
    (tmp_path/'harmonic_artists.json').write_text(json.dumps({
        'camera': [ax.azim, ax.elev], 'axes_position': list(ax.get_position().bounds),
        'colorbar_position': list(fig.axes[1].get_position().bounds),
        'field_rank': len(initial.pivots), 'grid': list(mappable.get_array().shape)
    }, indent=2)+'\n')
    plt.close(fig)


def test_bumpy_source_radius_colors_and_limits():
    field = ColorField()
    cmap = page._website_colormap()
    fig, ax, mappable = plot_sphere(field, projection='bumpy', n_pts=17,
                                    cmap=cmap, return_mappable=True)
    artist = ax.collections[0]
    ll, tt = np.meshgrid(np.linspace(-np.pi, np.pi, 17), np.linspace(0., np.pi, 17))
    values = field(ll, tt)
    radius = 1 + .15 * (2 * (values-values.min()) / (values.max()-values.min()) - 1)
    elevation = np.pi/2-tt
    expected = np.stack((radius*np.cos(elevation)*np.cos(ll),
                         radius*np.cos(elevation)*np.sin(ll), radius*np.sin(elevation)), axis=-1)
    np.testing.assert_array_equal(artist._source_vertices, expected)
    np.testing.assert_array_equal(mappable.get_array(), values)
    np.testing.assert_array_equal(artist._source_rgba,
                                  _matlab_facecolors(values, cmap, shade_data=expected[..., 2],
                                                     apply_lighting=True, norm=_normalize_values(values)))
    for limits in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()):
        np.testing.assert_array_equal(limits, (-1.15, 1.15))
    fig.canvas.draw()
    assert np.isfinite(artist._projected_triangles).all()
    plt.close(fig)


def test_explicit_surface_keywords_retain_existing_path():
    from unittest.mock import patch

    from mpl_toolkits.mplot3d.axes3d import Axes3D

    original = Axes3D.plot_surface
    calls = []

    def observe(ax, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(ax, *args, **kwargs)

    with patch.object(Axes3D, 'plot_surface', observe):
        fig, ax = plot_sphere(ColorField(), n_pts=9, alpha=.4, rasterized=True)
        fig.canvas.draw()
    assert len(calls) == 1
    assert calls[0]['rstride'] == calls[0]['cstride'] == 1
    assert calls[0]['shade'] is False and calls[0]['antialiased'] is True
    assert ax.collections[0].get_alpha() == .4
    assert ax.collections[0].get_rasterized()
    assert not isinstance(ax.collections[0], InterpolatedSphereSurface)
    plt.close(fig)


def test_equirectangular_projection_color_shape_and_limits():
    fig, ax, mappable = plot_sphere(ColorField(), projection='equirectangular',
                                    n_pts=17, clim=(-1., 1.5), return_mappable=True)
    assert mappable.get_array().shape == (17, 17)
    assert mappable.get_clim() == (-1., 1.5)
    assert not isinstance(ax.collections[0], InterpolatedSphereSurface)
    np.testing.assert_array_equal(ax.collections[0].get_array(), mappable.get_array())
    fig.canvas.draw()
    plt.close(fig)
