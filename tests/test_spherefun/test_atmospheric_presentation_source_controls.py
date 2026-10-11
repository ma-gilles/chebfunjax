# uses-numpy: independent palette, geometry and saved-image controls.
"""Bounded rendering-only source/default and measured website controls."""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from chebfunjax.plotting import (
    CHEBFUN_RC,
    PARULA,
    contour_sphere,
    matlab_explicit_camera,
    matlab_view,
    plot_sphere,
    save_chebfun_figure,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('atmospheric_page', ROOT/'examples/sphere/atmospherictemperature.py')
page = importlib.util.module_from_spec(spec)
sys.modules['atmospheric_page'] = page
spec.loader.exec_module(page)


class Field:
    def __call__(self, longitude, theta):
        return np.cos(np.asarray(theta)) + .2*np.sin(np.asarray(theta))*np.cos(np.asarray(longitude))

    def sample(self, longitude_count, theta_count):
        ll, tt = np.meshgrid(np.linspace(-np.pi, np.pi, longitude_count, endpoint=False),
                             np.linspace(0., np.pi, theta_count))
        return self(ll, tt)


def test_website_jet64_independent_channel_intervals():
    expected = np.zeros((64, 3))
    expected[24:40, 0] = np.arange(1, 17)/16
    expected[40:56, 0] = 1
    expected[56:, 0] = np.arange(15, 7, -1)/16
    expected[8:24, 1] = np.arange(1, 17)/16
    expected[24:40, 1] = 1
    expected[40:56, 1] = np.arange(15, -1, -1)/16
    expected[:8, 2] = np.arange(9, 17)/16
    expected[8:24, 2] = 1
    expected[24:40, 2] = np.arange(15, -1, -1)/16
    cmap = page._website_jet()
    assert cmap.N == 64
    np.testing.assert_array_equal(cmap.colors, expected)


def test_default_single_level_uses_source_endpoint_custom_compatibility():
    for options, expected in [({}, PARULA(1.)[:3]), ({'cmap': 'jet'}, 'k'),
                              ({'fmt': 'b-', 'linewidth': 2.}, 'b')]:
        fig, ax = contour_sphere(Field(), levels=[0., 0.], n_pts=17, **options)
        try:
            assert ax.lines
            for line in ax.lines:
                if isinstance(expected, str):
                    assert line.get_color() == expected
                else:
                    np.testing.assert_array_equal(line.get_color(), expected)
                if 'linewidth' in options:
                    assert line.get_linewidth() == 2.
            fig.canvas.draw()
        finally:
            plt.close(fig)


@pytest.mark.parametrize('style', [{}, CHEBFUN_RC,
                                  {'axes.linewidth': .1, 'axes.edgecolor': 'red'}],
                         ids=['default', 'example', 'foreign-style'])
def test_actual_artist_website_first_figure_layout(tmp_path, style):
    with mpl.rc_context(style):
        _assert_actual_artist_website_first_figure_layout(tmp_path)


def _assert_actual_artist_website_first_figure_layout(tmp_path):
    fig, ax, mappable = plot_sphere(Field(), n_pts=200, cmap=page._website_jet(),
                                    return_mappable=True)
    geometry = ax.collections[0]._source_vertices.copy()
    rgba = ax.collections[0]._source_rgba.copy()
    field = np.array(mappable.get_array(), copy=True)
    ax.set_axis_off()
    matlab_view(ax, 50, 0)
    matlab_explicit_camera(ax, **page._SOURCE_SURFACE_CAMERA)
    page._website_colorbar_layout(fig, ax, mappable)
    path = tmp_path/'analytic_website01.png'
    save_chebfun_figure(fig, path, size=(600, 270))
    np.testing.assert_array_equal(ax.collections[0]._source_vertices, geometry)
    np.testing.assert_array_equal(ax.collections[0]._source_rgba, rgba)
    np.testing.assert_array_equal(mappable.get_array(), field)
    np.testing.assert_allclose(fig.axes[1].get_position().bounds,
                               (477/600, 31/270, 38/600, 219/270), rtol=0, atol=1e-15)
    with Image.open(path) as image:
        assert image.size == (600, 270)
        rgb = np.asarray(image.convert('RGB'))
    # Black frame endpoints are inclusive raster coordinates, unlike widths.
    dark = np.max(rgb, axis=2) < 60
    columns = [x for x in range(460, 525) if dark[25:235, x].sum() > 200]
    rows = [y for y in range(10, 250) if dark[y, 479:514].sum() > 30]
    assert columns == [477, 515]
    assert rows == [20, 239]
    yy, xx = np.nonzero(np.ptp(rgb[:, :420].astype(int), axis=-1) > 32)
    bounds = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
    (tmp_path/'geometry_observed.json').write_text(json.dumps({'sphere_bounds': bounds,
        'reference_sphere_bounds': [128, 20, 348, 240],
        'axes': list(ax.get_position().bounds),
        'colorbar': list(fig.axes[1].get_position().bounds)}, indent=2)+'\n')
    assert bounds == [128, 20, 348, 240]
    plt.close(fig)
