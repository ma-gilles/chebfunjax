"""Source-literal geometry, coefficient floor, and colors for plotcoeffs."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3t import Chebfun3T
from chebfunjax.plotting import PARULA


def test_plotcoeffs_uses_ndgrid_f_order_log_coefficients_and_native_axes():
    coeffs = jnp.asarray(
        [
            [[0.0, 4.0], [2.0, 0.0], [8.0, 16.0]],
            [[1.0, 0.0], [0.5, 32.0], [64.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    tensor = Chebfun3T.from_coeffs(coeffs, vscale=2.0)
    fig, ax = tensor.plotcoeffs()
    try:
        artist = ax.collections[0]
        x, y, z = artist._offsets3d
        np.testing.assert_array_equal(x, [1, 2] * 6)
        np.testing.assert_array_equal(y, [1, 1, 2, 2, 3, 3] * 2)
        np.testing.assert_array_equal(z, [1] * 6 + [2] * 6)

        eps = np.finfo(np.float64).eps
        magnitudes_f = np.asarray(coeffs).ravel(order="F")
        magnitudes_f[magnitudes_f == 0] = 2.0 * eps
        expected_cdata = np.log10(magnitudes_f)
        np.testing.assert_allclose(
            np.asarray(artist.get_array()), expected_cdata, rtol=2e-16, atol=0
        )
        fig.canvas.draw()
        np.testing.assert_array_equal(
            artist.get_facecolors()[:, 3], np.ones(expected_cdata.size)
        )
        assert artist.get_cmap() is PARULA
        assert artist.get_depthshade() is False
        assert np.asarray(artist.get_edgecolors()).shape == (0, 4)
        np.testing.assert_array_equal(artist.get_linewidths(), [0.5])
        assert len(fig.axes) == 2
        assert fig.axes[1].get_label() == "<colorbar>"
        assert ax._draw_grid is True
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect((1, 1, 1))
        ax.get_proj()
        azimuth = np.deg2rad(-30.0)
        elevation = np.deg2rad(35.0)
        expected_camera_direction = np.asarray(
            [
                np.cos(elevation) * np.sin(azimuth),
                -np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation),
            ]
        )
        np.testing.assert_allclose(
            ax._view_w, expected_camera_direction, rtol=0, atol=2e-15
        )
    finally:
        plt.close(fig)


def test_plotcoeffs_zero_vscale_uses_eps_floor():
    tensor = Chebfun3T.from_coeffs(jnp.zeros((2, 1, 1)))
    fig, ax = tensor.plotcoeffs()
    try:
        artist = ax.collections[0]
        np.testing.assert_allclose(
            np.asarray(artist.get_array()),
            np.full(2, np.log10(np.finfo(np.float64).eps)),
            rtol=2e-16,
            atol=0,
        )
    finally:
        plt.close(fig)


def test_plotcoeffs_empty_uses_empty_plot_without_colorbar():
    tensor = Chebfun3T.empty()
    fig, ax = tensor.plotcoeffs()
    try:
        assert len(ax.lines) == 1
        assert len(fig.axes) == 1
        assert not ax.collections
    finally:
        plt.close(fig)


def test_plotcoeffs_integer_coefficients_promote_before_epsilon_floor():
    tensor = Chebfun3T.from_coeffs(
        jnp.asarray([[[0]], [[1]]], dtype=jnp.int32), vscale=2
    )
    fig, ax = tensor.plotcoeffs()
    try:
        actual = np.asarray(ax.collections[0].get_array())
        expected = np.asarray(
            [np.log10(2 * np.finfo(np.float64).eps), 0.0]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-16, atol=0)
    finally:
        plt.close(fig)
