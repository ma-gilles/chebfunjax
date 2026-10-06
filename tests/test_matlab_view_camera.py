"""Independent camera controls; no coast, visibility helper or raster image.

Provenance
----------
MATLAB source : view (MathWorks camera line-of-sight convention)
Chebfun commit: 7574c77
Source AtmosphericTemperature calls view([50 0]) and view([50 5]). These are
new independent controls, not an original upstream MATLAB test port.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# uses-numpy: Independent test geometry and Matplotlib interface.
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

from chebfunjax.plotting import matlab_view  # noqa: E402


@pytest.mark.parametrize('azimuth,elevation,front', [
    (0., 0., [0., -1., 0.]),
    (90., 0., [1., 0., 0.]),
    (180., 0., [0., 1., 0.]),
    (270., 0., [-1., 0., 0.]),
    (0., 90., [0., 0., 1.]),
    (0., -90., [0., 0., -1.]),
])
@pytest.mark.parametrize('projection', ['ortho', 'persp'])
def test_cardinals_project_expected_front_point_closer(azimuth, elevation, front, projection):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection='3d')
        ax.set(xlim=(-1., 1.), ylim=(-1., 1.), zlim=(-1., 1.))
        ax.set_box_aspect([1., 1., 1.])
        ax.set_proj_type(projection)
        assert matlab_view(ax, azimuth, elevation) is ax
        xyz = np.asarray([front, -np.asarray(front)])
        x, y, depth = proj3d.proj_transform(*xyz.T, ax.get_proj())
        # Matplotlib's actual projection places the eye-facing endpoint nearer
        # at smaller projected depth; both endpoints lie on its central ray.
        assert depth[0] < depth[1]
        np.testing.assert_allclose([x[0], y[0]], [x[1], y[1]], rtol=0, atol=2e-14)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('elevation', [0., 5.])
def test_actual_source_angles_keep_projection_zoom_and_layout(elevation):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection='3d')
        ax.set_proj_type('persp', focal_length=2.5)
        ax.set_box_aspect([1., 2., 3.], zoom=.7)
        aspect = ax._box_aspect.copy()
        position = ax.get_position().bounds
        limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
        matlab_view(ax, 50., elevation)
        assert ax.azim == -40.
        assert ax.elev == elevation
        assert ax._focal_length == 2.5
        np.testing.assert_array_equal(ax._box_aspect, aspect)
        assert ax.get_position().bounds == position
        assert (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) == limits
    finally:
        plt.close(fig)
