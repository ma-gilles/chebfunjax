"""R2024b material and source-light controls for Random2D surface rendering.

The fixture captures the properties of MATLAB surf(xx,yy,C,...) after the
Chebfun defaults and camlight left. Arithmetic controls below use the captured
material values and simple geometry with literal expected colors. They do not
claim native rasterization or pixel parity.
"""
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax._surface_lighting import (
    matlab_camera_and_camlight_left,
    matlab_flat_face_lighting,
)

_CAPTURE = Path(__file__).parent / "fixtures" / "random2d_surf_material_r2024b.json"


def test_fixture_binds_chebfun_surface_and_material_defaults():
    capture = json.loads(_CAPTURE.read_text())
    assert capture["matlab_release"] == "2024b"
    assert capture["source_call"] == (
        "surf(xx,yy,C,'FaceColor','interp','EdgeAlpha',.5,'EdgeColor','none')")
    surface = capture["surface"]
    assert surface["FaceLighting"] == "flat"
    assert surface["BackFaceLighting"] == "reverselit"
    assert surface["AmbientStrength"] == 0.3
    assert surface["DiffuseStrength"] == 0.6
    assert surface["SpecularStrength"] == 0.9
    assert surface["SpecularExponent"] == 10
    assert surface["EdgeColor"] == "none"
    assert surface["EdgeAlpha"] == 0.5
    assert capture["light_style"] == "local"
    np.testing.assert_array_equal(capture["light_color"], [1, 1, 1])


def test_left_camera_and_light_positions_follow_captured_basis():
    camera, light = matlab_camera_and_camlight_left(
        -20.0, 50.0, jnp.array([1., .5, .5]), jnp.array([2., 1., 3.]),
        radius=10.)
    az, el = np.deg2rad([-20., 50.])
    eye = np.array([np.sin(az)*np.cos(el),
                    -np.cos(az)*np.cos(el), np.sin(el)])
    right = np.array([np.cos(az), np.sin(az), 0.])
    up = np.array([-np.sin(az)*np.sin(el),
                   np.cos(az)*np.sin(el), np.cos(el)])
    target = np.array([1., .5, .5])
    dar = np.array([2., 1., 3.])
    expected_camera = target + 10*dar*eye
    light_direction = -np.sqrt(3)/4*right + .5*up + .75*eye
    expected_light = target + 10*dar*light_direction
    np.testing.assert_allclose(camera, expected_camera, rtol=0, atol=2e-14)
    np.testing.assert_allclose(light, expected_light, rtol=0, atol=2e-14)
    capture = json.loads(_CAPTURE.read_text())
    np.testing.assert_allclose(
        np.asarray(capture["light_position"]),
        [-6.6345726654530885, .9763584978526507, 23.776869822197405],
        rtol=0, atol=1e-13)


def test_flat_local_phong_face_matches_literal_material_control_and_jit():
    x = jnp.array([[0., 1.], [0., 1.]])
    y = jnp.array([[0., 0.], [1., 1.]])
    z = jnp.zeros((2, 2))
    rgba = jnp.broadcast_to(jnp.array([.2, .4, .6, 1.]), (2, 2, 4))
    camera = jnp.array([.5, .5, 1.])
    light = jnp.array([1.5, .5, 1.])
    eager = matlab_flat_face_lighting(
        rgba, x, y, z, camera, light, jnp.ones(3), ambient=.3, diffuse=.6,
        specular=.9, specular_exponent=10., backface="reverselit")
    ndotl = np.sqrt(.5)
    ndoth = np.sqrt((1 + ndotl) / 2)
    illumination = .3 + .6*ndotl
    highlight = .9*ndoth**10
    expected = np.array([[[.2*illumination+highlight,
                           .4*illumination+highlight,
                           .6*illumination+highlight, 1.]]])
    np.testing.assert_allclose(eager, expected, rtol=0, atol=2e-15)
    compiled = jax.jit(matlab_flat_face_lighting, static_argnames=("backface",))(
        rgba, x, y, z, camera, light, jnp.ones(3), ambient=.3, diffuse=.6,
        specular=.9, specular_exponent=10., backface="reverselit")
    np.testing.assert_allclose(compiled, expected, rtol=0, atol=2e-15)


def test_flat_face_grid_shape_and_backface_reversal():
    x = jnp.tile(jnp.arange(4., dtype=jnp.float64), (3, 1))
    y = jnp.tile(jnp.arange(3., dtype=jnp.float64)[:, None], (1, 4))
    z = jnp.zeros((3, 4))
    rgba = jnp.ones((3, 4, 4)) * jnp.array([.1, .2, .3, 1.])
    camera = jnp.array([1.5, 1., 2.])
    light = jnp.array([1.5, 1., 2.])
    got = matlab_flat_face_lighting(
        rgba, x, y, z, camera, light, jnp.ones(3), backface="reverselit")
    assert got.shape == (2, 3, 4)
    assert bool(jnp.all(jnp.isfinite(got)))
    front = matlab_flat_face_lighting(
        rgba, x, y, z, camera, light, jnp.ones(3), backface="lit")
    np.testing.assert_allclose(got, front, rtol=0, atol=1e-15)


def test_matlab_light_tracks_actual_axes3d_projection_geometry():
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set_xlim(0., 2.)
        ax.set_ylim(0., 1.)
        ax.set_zlim(-1., 2.)
        ax.view_init(elev=50., azim=-110.)
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        target = np.mean(limits, axis=1)
        dar = np.ptp(limits, axis=1)/np.asarray(ax.get_box_aspect())
        camera, light = matlab_camera_and_camlight_left(
            ax.azim+90., ax.elev, target, dar, radius=ax._dist)
        az, el = np.deg2rad([ax.azim, ax.elev])
        mpl_eye = np.array([np.cos(el)*np.cos(az),
                            np.cos(el)*np.sin(az), np.sin(el)])
        scaled_eye = (np.asarray(camera)-target)/dar/ax._dist
        np.testing.assert_allclose(scaled_eye, mpl_eye, rtol=0, atol=2e-15)
        ray = (np.asarray(light)-target)/dar/ax._dist
        right = np.array([np.cos(np.deg2rad(ax.azim+90.)),
                          np.sin(np.deg2rad(ax.azim+90.)), 0.])
        up = np.cross(mpl_eye, right)
        np.testing.assert_allclose(
            [ray@right, ray@up, ray@mpl_eye],
            [-np.sqrt(3)/4, .5, .75], rtol=0, atol=2e-15)
    finally:
        plt.close(fig)
