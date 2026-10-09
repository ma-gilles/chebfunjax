"""Independent source-grid and actual colorbar/artist snapshot controls."""

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# uses-numpy: independent analytic grid and Matplotlib color inspection.
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import LightSource, Normalize  # noqa: E402

from chebfunjax.plotting import plot_sphere  # noqa: E402


class AnalyticField:
    def __init__(self, offset=0.):
        self.calls = []
        self.offset = offset

    def __call__(self, longitude, theta):
        self.calls.append((longitude.shape, theta.shape))
        return self.offset + jnp.cos(theta) + .25 * jnp.sin(theta) * jnp.cos(longitude)


def independent_grid(n, offset=0.):
    longitude, theta = np.meshgrid(np.linspace(-np.pi, np.pi, n), np.linspace(0, np.pi, n))
    return offset + np.cos(theta) + .25 * np.sin(theta) * np.cos(longitude)


def test_default_source_grid_one_evaluation_and_colorbar_limits():
    field = AnalyticField()
    fig, ax, mappable = plot_sphere(field, return_mappable=True)
    try:
        expected = independent_grid(200)
        assert field.calls == [((200, 200), (200, 200))]
        np.testing.assert_allclose(mappable.get_array(), expected, rtol=0, atol=8e-16)
        np.testing.assert_allclose(mappable.get_clim(), [expected.min(), expected.max()], rtol=0, atol=8e-16)
        # A different sampling grid must not supply the normalization.
        assert abs(mappable.norm.vmax - independent_grid(12).max()) > 1e-5
        surface = ax.collections[0]
        fig.canvas.draw()
        before = np.array(surface.get_facecolor(), copy=True)
        colorbar = fig.colorbar(mappable, ax=ax)
        fig.canvas.draw()
        np.testing.assert_array_equal(surface.get_facecolor(), before)
        assert colorbar.mappable is mappable and colorbar.norm is mappable.norm
        # New adapter control: exact colorbar-edge equality is not a source
        # requirement. Fresh MATLAB/source-grid CLim upper=1.030738760547388;
        # Matplotlib's inverse-normalization edge=1.0307387605473879 (one ULP).
        # Subtraction then addition can round: gamma_2 bounds those two
        # operations, independently of their implementation. Existing
        # source-grid and CLim bounds above remain unchanged.
        limits = np.asarray(mappable.get_clim())
        unit_roundoff = np.finfo(float).eps / 2
        gamma2 = 2 * unit_roundoff / (1 - 2 * unit_roundoff)
        edge_bound = gamma2 * np.sum(np.abs(limits))
        assert np.max(np.abs(np.asarray([colorbar.vmin, colorbar.vmax]) - limits)) <= edge_bound
        assert len(field.calls) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("projection", ["sphere", "bumpy", "equirectangular"])
def test_cmap_and_snapshot_mutations_do_not_recolor_existing_artist(projection):
    fig, ax, mappable = plot_sphere(AnalyticField(), n_pts=12, cmap="plasma",
                                   projection=projection, return_mappable=True)
    try:
        expected = independent_grid(12)
        assert mappable.cmap.name == "plasma"
        np.testing.assert_allclose(mappable.to_rgba(expected),
                                   plt.get_cmap("plasma")(Normalize(expected.min(), expected.max())(expected)),
                                   rtol=0, atol=8e-16)
        fig.canvas.draw()
        artist = ax.collections[0]
        before = np.array(artist.get_facecolor(), copy=True)
        mappable.set_clim(-10, 20)
        mappable.set_cmap("viridis")
        mappable.set_array(np.zeros((3, 3)))
        fig.canvas.draw()
        np.testing.assert_array_equal(artist.get_facecolor(), before)
    finally:
        plt.close(fig)


def test_constant_correction_and_existing_constant_norm_fallback():
    calls = []

    def almost_constant(longitude, theta):
        calls.append(longitude.shape)
        return 3. + 1e-13 * jnp.sin(theta) * jnp.cos(longitude)

    fig, _, mappable = plot_sphere(almost_constant, n_pts=12, return_mappable=True)
    try:
        assert calls == [(12, 12)]
        np.testing.assert_array_equal(mappable.get_array(), np.full((12, 12), 3.))
        assert mappable.get_clim() == (3., 4.)
    finally:
        plt.close(fig)


def test_multiple_surfaces_keep_independent_scales_and_default_pair_return():
    result = plot_sphere(AnalyticField(), n_pts=12)
    assert len(result) == 2
    fig, ax = result
    try:
        fig2, ax2, first = plot_sphere(AnalyticField(5), ax=ax, n_pts=12, return_mappable=True)
        _, _, second = plot_sphere(AnalyticField(20), ax=ax, n_pts=12, return_mappable=True)
        assert fig2 is fig and ax2 is ax
        np.testing.assert_allclose(np.array(second.get_clim()) - first.get_clim(), [15., 15.], rtol=0, atol=4e-15)
        assert first.norm is not second.norm
        assert len(ax.collections) == 3
    finally:
        plt.close(fig)


def test_bumpy_lighting_receives_same_scalar_colors(monkeypatch):
    # Observe the real host lighting call and preserve its behavior. Compare
    # its input against an independent grid/cmap calculation, not a helper.
    inputs = []
    original = LightSource.shade_rgb

    def observe(self, rgb, elevation, *args, **kwargs):
        inputs.append(np.array(rgb, copy=True))
        return original(self, rgb, elevation, *args, **kwargs)

    monkeypatch.setattr(LightSource, "shade_rgb", observe)
    fig, _, mappable = plot_sphere(AnalyticField(), n_pts=12, cmap="plasma",
                                  projection="bumpy", return_mappable=True)
    try:
        expected = independent_grid(12)
        assert len(inputs) == 1
        np.testing.assert_allclose(inputs[0], mappable.to_rgba(expected)[..., :3], rtol=0, atol=8e-16)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("projection", ["sphere", "equirectangular"])
def test_fixed_source_color_limits_reach_artist_and_colorbar(monkeypatch, projection):
    from mpl_toolkits.mplot3d import Axes3D

    colors = []
    original = Axes3D.plot_surface

    def observe(self, *args, **kwargs):
        colors.append(np.array(kwargs['facecolors'], copy=True))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes3D, 'plot_surface', observe)
    fig, ax, mappable = plot_sphere(AnalyticField(), n_pts=12, cmap="jet",
                                   projection=projection, clim=(-.5, 1.),
                                   return_mappable=True)
    try:
        expected = independent_grid(12)
        expected_rgba = plt.get_cmap("jet")(Normalize(-.5, 1.)(expected))
        assert mappable.get_clim() == (-.5, 1.)
        np.testing.assert_allclose(mappable.to_rgba(expected), expected_rgba,
                                   rtol=0, atol=8e-16)
        if projection == "sphere":
            assert len(colors) == 1
            np.testing.assert_allclose(colors[0], expected_rgba, rtol=0, atol=8e-16)
        else:
            assert ax.collections[0].get_clim() == (-.5, 1.)
        colorbar = fig.colorbar(mappable, ax=ax)
        assert colorbar.norm.vmin == -.5 and colorbar.norm.vmax == 1.
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("clim", [(1., 1.), (0., np.nan), (0., 1., 2.)])
def test_invalid_color_limits_rejected_before_sampling(clim):
    field = AnalyticField()
    with pytest.raises(ValueError, match="clim"):
        plot_sphere(field, clim=clim)
    assert field.calls == []
