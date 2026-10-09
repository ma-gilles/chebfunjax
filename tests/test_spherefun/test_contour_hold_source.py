"""Native spherefun/contour.m hold branch preserves an existing surface."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun


def test_held_contour_preserves_surface_and_axes():
    f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
    fig, ax = f.plot(n_pts=24, cmap='jet')
    try:
        ax.view_init(elev=31, azim=57)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-4, 4)
        ax.set_box_aspect((2, 3, 4))
        collections = tuple(ax.collections)
        colors = [np.array(a.get_facecolors(), copy=True) for a in collections]
        bounds = ax.get_position().bounds
        aspect = ax.get_box_aspect().copy()
        out_fig, out_ax = f.contour(ax=ax, hold=True, n_pts=24,
                                   levels=[0., 0.], fmt='k-')
        assert out_fig is fig and out_ax is ax
        assert tuple(ax.collections) == collections
        for artist, before in zip(collections, colors):
            np.testing.assert_array_equal(artist.get_facecolors(), before)
        assert (ax.elev, ax.azim) == (31, 57)
        assert ax.get_xlim() == (-2, 2)
        assert ax.get_ylim() == (-3, 3)
        assert ax.get_zlim() == (-4, 4)
        assert ax.get_position().bounds == bounds
        np.testing.assert_array_equal(ax.get_box_aspect(), aspect)
        assert len(ax.lines) > 0
        for line in ax.lines:
            xx, yy, zz = line.get_data_3d()
            assert line.get_color() == 'k'
            # Same geometric allowance as existing equator controls.
            np.testing.assert_allclose(zz, 0, atol=5e-14, rtol=0)
            np.testing.assert_allclose(np.asarray(xx)**2+np.asarray(yy)**2, 1,
                                       atol=5e-14, rtol=0)
    finally:
        plt.close(fig)


def test_unheld_contour_draws_background():
    f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
    fig, ax = f.contour(n_pts=24, levels=[0., 0.])
    try:
        assert len(ax.collections) == 1
        assert len(ax.lines) > 0
    finally:
        plt.close(fig)
