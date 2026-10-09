"""Unit-sphere contour framing: native contour.m has no overflowing viewport."""
# uses-numpy: independent geometry and Matplotlib artist observations.
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

from chebfunjax.plotting import (
    contour_sphere,
    matlab_explicit_camera,
    matlab_view,
    plot_earth,
    save_chebfun_figure,
)
from chebfunjax.spherefun.spherefun import Spherefun


def field():
    lon = jnp.linspace(-jnp.pi, jnp.pi, 17)[:-1]
    lat = jnp.linspace(0., jnp.pi, 17)
    ll, tt = jnp.meshgrid(lon, lat)
    return Spherefun.from_values(jnp.cos(tt)+.25*jnp.sin(tt)*jnp.cos(ll))


def test_new_contour_axes_use_subplot_without_canvas_overflow():
    fig, ax = contour_sphere(field(), n_pts=33, levels=[-.5, 0., .5])
    try:
        assert ax.get_subplotspec() is not None
        left, bottom, width, height = ax.get_position(original=True).bounds
        allowance = 64*np.finfo(float).eps
        assert left >= -allowance and bottom >= -allowance
        assert left+width <= 1+allowance and bottom+height <= 1+allowance
        assert not hasattr(ax, '_matlab_explicit_camera')
    finally:
        plt.close(fig)


def test_actual_contour_camera_layout_stays_inside_canvas_preserves_geometry(tmp_path):
    fig, ax = contour_sphere(field(), n_pts=33, levels=[-.5, 0., .5], linewidth=2.)
    try:
        ax.set_axis_off()
        matlab_view(ax, 50., 5.)
        metadata = json.loads((Path(__file__).parent/'fixtures/sphere_explicit_camera_metadata.json').read_text())['cases']['contour']
        matlab_explicit_camera(ax, position=metadata['CameraPosition'],
                               target=metadata['CameraTarget'], up=metadata['CameraUpVector'],
                               view_angle=metadata['CameraViewAngle'], data_aspect=metadata['DataAspectRatio'])
        plot_earth(ax, 'k-')
        before = [tuple(np.array(c, copy=True) for c in line.get_data_3d()) for line in ax.lines]
        path = tmp_path/'contour_viewport.png'
        save_chebfun_figure(fig, path, size=(600, 270), layout='matlab')
        assert Image.open(path).size == (600, 270)
        for line, expected in zip(ax.lines, before):
            for actual, coordinate in zip(line.get_data_3d(), expected):
                np.testing.assert_array_equal(actual, coordinate)
        # Extremal unit-sphere points in the camera plane are independent of
        # the contour extraction. They bound the entire physical sphere.
        az, el = np.deg2rad(metadata['View'])
        right = np.array([np.cos(az), np.sin(az), 0.])
        up = np.array([-np.sin(el)*np.sin(az), np.sin(el)*np.cos(az), np.cos(el)])
        points = np.stack([right, -right, up, -up])
        x, y, _ = proj3d.proj_transform(*points.T, ax.get_proj())
        pixels = ax.transData.transform(np.stack([x, y], axis=1))
        allowance = 64*np.finfo(float).eps*max(fig.bbox.width, fig.bbox.height, 1.)
        assert np.min(pixels[:, 0]) >= -allowance
        assert np.max(pixels[:, 0]) <= fig.bbox.width+allowance
        assert np.min(pixels[:, 1]) >= -allowance
        assert np.max(pixels[:, 1]) <= fig.bbox.height+allowance
    finally:
        plt.close(fig)


def test_supplied_contour_axes_keep_identity_without_implicit_camera():
    fig = plt.figure()
    ax = fig.add_axes([.2, .2, .5, .5], projection='3d')
    try:
        result_fig, result_ax = contour_sphere(field(), ax=ax, n_pts=17, levels=[0., 0.])
        assert result_fig is fig and result_ax is ax
        assert ax.get_subplotspec() is None
        assert not hasattr(ax, '_matlab_explicit_camera')
    finally:
        plt.close(fig)
