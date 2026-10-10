"""Captured-camera controls; equations from licensed camzoom/viewmtx sources."""
# uses-numpy: independent graphics/landmark oracle and raster observations.
import json
import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

from chebfunjax.plotting import (
    _sphere_visible_vertices,
    matlab_explicit_camera,
    matlab_view,
    plot_earth,
    plot_sphere,
    save_chebfun_figure,
)

FIXTURE = json.loads((Path(__file__).parent/'fixtures/sphere_explicit_camera_metadata.json').read_text())


def settings(case):
    a = FIXTURE['cases'][case]
    return dict(position=a['CameraPosition'], target=a['CameraTarget'],
                up=a['CameraUpVector'], view_angle=a['CameraViewAngle'],
                data_aspect=a['DataAspectRatio'])


def oracle(ax, case, points):
    # Independent closed-form licensed viewmtx rows, not candidate cross products.
    a = FIXTURE['cases'][case]
    az, el = map(math.radians, a['View'])
    right = np.array([math.cos(az), math.sin(az), 0.])
    upward = np.array([-math.sin(el)*math.sin(az), math.sin(el)*math.cos(az), math.cos(el)])
    distance = math.sqrt(sum(((x-y)/d)**2 for x, y, d in zip(a['CameraPosition'], a['CameraTarget'], a['DataAspectRatio'])))
    fov = 2*distance*math.tan(math.radians(a['CameraViewAngle'])/2)
    center = np.array([ax.bbox.x0+ax.bbox.width/2, ax.bbox.y0+ax.bbox.height/2])
    return center + np.stack([points@right, points@upward], axis=1)*ax.bbox.height/fov


def pixels(ax, points):
    x, y, _ = proj3d.proj_transform(*points.T, ax.get_proj())
    return ax.transData.transform(np.stack((x, y), axis=1))


def make_axes():
    fig = plt.figure(figsize=(6, 2.7), dpi=100)
    ax = fig.add_axes([.13, .11, .775, .815], projection='3d')
    ax.set(xlim=(-1., 1.), ylim=(-1., 1.), zlim=(-1., 1.))
    ax.set_box_aspect((1., 1., 1.))
    ax.set_axis_off()
    return fig, ax


@pytest.mark.parametrize('case', ['surface', 'contour'])
def test_captured_landmarks_match_independent_source_equations(case):
    fig, ax = make_axes()
    try:
        points = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
                           [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
        result = matlab_explicit_camera(ax, **settings(case))
        fig.canvas.draw()
        allowance = 64*np.finfo(float).eps*max(fig.bbox.width, fig.bbox.height, 1.)
        np.testing.assert_allclose(pixels(ax, points), oracle(ax, case, points), rtol=0, atol=allowance)
        assert result['projection'] == 'orthographic' and ax._focal_length == math.inf
        assert tuple(ax.get_xlim()) == (-1., 1.)
        center = pixels(ax, points[:1])[0]
        np.testing.assert_allclose(center, [ax.bbox.x0+ax.bbox.width/2, ax.bbox.y0+ax.bbox.height/2], rtol=0, atol=allowance)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('case', ['surface', 'contour'])
def test_layout_resize_recomputes_camera_viewport(case, tmp_path):
    fig, ax = make_axes()
    try:
        points = np.eye(3)
        matlab_explicit_camera(ax, **settings(case))
        for size in [(600, 270), (800, 320)]:
            ax.set_title('Late annotation')
            path = tmp_path/f'{case}_{size[0]}.png'
            save_chebfun_figure(fig, path, size=size, layout='matlab')
            assert Image.open(path).size == size
            # save restores DPI, so draw at the current size before comparing.
            fig.canvas.draw()
            allowance = 64*np.finfo(float).eps*max(fig.bbox.width, fig.bbox.height, 1.)
            np.testing.assert_allclose(pixels(ax, points), oracle(ax, case, points), rtol=0, atol=allowance)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('case', ['surface', 'contour'])
def test_inverse_projection_column_preserves_camera_ray_and_depth(case):
    fig, ax = make_axes()
    try:
        source = settings(case)
        matlab_explicit_camera(ax, **source)
        fig.canvas.draw()
        direction = np.array(source['position'])-np.array(source['target'])
        direction /= np.linalg.norm(direction)
        matrix = ax.get_proj()
        inverse_column = np.linalg.inv(matrix)[:, 2]
        assert inverse_column[3] == 0.
        observed = -inverse_column[:3]/np.linalg.norm(inverse_column[:3])
        np.testing.assert_allclose(observed, direction, rtol=0, atol=64*np.finfo(float).eps)
        points = np.stack([direction, -direction])
        _, _, depth = proj3d.proj_transform(*points.T, matrix)
        assert depth[0] < depth[1]
        np.testing.assert_array_equal(_sphere_visible_vertices(jnp.asarray(points), matrix, 1.), [True, False])
    finally:
        plt.close(fig)


def test_actual_sphere_coast_camera_preserves_source_geometry(tmp_path):
    fig, ax = plot_sphere(lambda l, t: np.cos(np.asarray(t)), n_pts=17, cmap='gray')
    try:
        ax.set_axis_off()
        surface = ax.collections[0]
        geometry = _surface_geometry(surface).copy()
        (coast,) = plot_earth(ax, 'r-', linewidth=2.)
        original = np.array(coast.get_data_3d(), copy=True)
        matlab_explicit_camera(ax, **settings('surface'))
        ax.set_title('Explicit captured camera')
        save_chebfun_figure(fig, tmp_path/'explicit_surface.png', size=(600, 270), layout='matlab')
        np.testing.assert_array_equal(_surface_geometry(surface), geometry)
        np.testing.assert_array_equal(coast.get_data_3d(), original)
        rgba = np.asarray(fig.canvas.buffer_rgba())
        red = (rgba[..., 0] > 150) & (rgba[..., 1] < 100) & (rgba[..., 2] < 100)
        assert red.any()
    finally:
        plt.close(fig)


def test_supplied_axes_and_angle_only_adapter_do_not_enable_policy():
    fig, ax = make_axes()
    try:
        ax.set_proj_type('persp', focal_length=2.)
        plot_sphere(lambda l, t: np.cos(np.asarray(t)), ax=ax, n_pts=9)
        assert not hasattr(ax, '_matlab_explicit_camera')
        matlab_view(ax, 50., 5.)
        assert ax._focal_length == 2.
        assert ax.azim == -40. and ax.elev == 5.
    finally:
        plt.close(fig)


@pytest.mark.parametrize('change', ['view', 'projection', 'zoom'])
def test_later_explicit_user_camera_change_opts_out(change):
    fig, ax = make_axes()
    try:
        matlab_explicit_camera(ax, **settings('surface'))
        original = ax._matlab_explicit_camera['original_get_proj']
        if change == 'view':
            matlab_view(ax, 20., 30.)
        elif change == 'projection':
            ax.set_proj_type('persp')
        else:
            ax.set_box_aspect((1., 1., 1.), zoom=1.2)
        np.testing.assert_array_equal(ax.get_proj(), original())
    finally:
        plt.close(fig)


def test_invalid_camera_rejected_before_axes_mutation():
    fig, ax = make_axes()
    try:
        before = ax.get_proj().copy()
        for invalid in [dict(position=(0., 0., 0.)), dict(up=(0., 0., 0.)),
                        dict(view_angle=0.), dict(data_aspect=(0., 1., 1.))]:
            kwargs = settings('surface') | invalid
            with pytest.raises(ValueError):
                matlab_explicit_camera(ax, **kwargs)
        np.testing.assert_array_equal(ax.get_proj(), before)
        assert not hasattr(ax, '_matlab_explicit_camera')
    finally:
        plt.close(fig)


def test_reapplying_explicit_camera_is_idempotent():
    fig, ax = make_axes()
    try:
        matlab_explicit_camera(ax, **settings('surface'))
        fig.canvas.draw()
        before = ax.get_proj().copy()
        matlab_explicit_camera(ax, **settings('surface'))
        fig.canvas.draw()
        np.testing.assert_array_equal(ax.get_proj(), before)
    finally:
        plt.close(fig)


def _surface_geometry(artist):
    # Gouraud artist retains source vertices; legacy polygons retain _vec.
    geometry = artist._source_vertices if hasattr(artist, "_source_vertices") else artist._vec
    assert geometry.size > 0
    return geometry
