"""Source surface properties and physical-coordinate tests, not PNG fitting."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import proj3d

from chebfunjax import plotting as p
from chebfunjax._colormaps import _PARULA_R2018A, parula_colormap
from chebfunjax._sphere_surface import InterpolatedSurface


class Field:
    def __init__(self, function, domain=(-2., 2., -1., 1.)):
        self.function, self.domain = function, domain

    def __call__(self, x, y):
        return self.function(x, y)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def projected_pixels(ax, vertices):
    q = np.asarray(vertices)
    x, y, _ = proj3d.proj_transform(q[:, 0], q[:, 1], q[:, 2], ax.get_proj())
    return ax.transData.transform(np.column_stack((x, y)))


def test_default_palette_is_full_exact_table():
    assert p.PARULA.N == 256
    np.testing.assert_array_equal(p.PARULA.colors, _PARULA_R2018A)
    explicit64 = parula_colormap(64)
    assert explicit64.N == 64
    np.testing.assert_array_equal(explicit64.colors[0], _PARULA_R2018A[0])
    np.testing.assert_array_equal(explicit64.colors[-1], _PARULA_R2018A[-1])


@pytest.mark.parametrize("form", ["height", "color", "parametric"])
def test_surf_preserves_sampled_vertex_colors_and_coordinates(form):
    x = Field(lambda x, y: x)
    y = Field(lambda x, y: y)
    z = Field(lambda x, y: x + 2*y)
    color = Field(lambda x, y: y - x)
    args = (x, y, z) if form == "parametric" else ((z, color) if form == "color" else (z,))
    fig, ax = p.surf(*args, n_pts=7)
    artist = ax.collections[0]
    assert isinstance(artist, InterpolatedSurface)
    xx, yy = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-1, 1, 7))
    zz = xx + 2*yy
    np.testing.assert_allclose(artist._source_vertices, np.stack((xx, yy, zz), axis=-1))
    cc = yy-xx if form == "color" else zz
    np.testing.assert_array_equal(artist._source_rgba, p.PARULA(p._normalize_values(cc)(cc)))
    assert artist.get_edgecolors().size == 0
    fig.canvas.draw()


def test_default_sampling_and_domain_limits():
    _, ax = p.surf(Field(lambda x, y: x+y))
    assert ax.collections[0]._source_vertices.shape == (200, 200, 3)
    np.testing.assert_array_equal(ax.get_xlim(), (-2, 2))
    np.testing.assert_array_equal(ax.get_ylim(), (-1, 1))


@pytest.mark.parametrize("figsize", [(6., 2.7), (2.7, 6.)])
def test_topdown_equal_units_and_viewport_fit_survive_resize(figsize):
    fig, ax = p.surf(Field(lambda x, y: x+y), n_pts=9)
    p.matlab_view(ax, 0, 90)
    p.matlab_axis_equal(ax)
    for size in (figsize, figsize[::-1]):
        fig.set_size_inches(*size)
        fig.canvas.draw()
        points = projected_pixels(ax, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        lengths = np.linalg.norm(points[1:]-points[0], axis=1)
        np.testing.assert_allclose(lengths[0], lengths[1], rtol=2e-14)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        corners = projected_pixels(ax, [[xlim[0], ylim[0], 0], [xlim[1], ylim[1], 0]])
        spans = np.abs(corners[1]-corners[0])
        np.testing.assert_allclose(spans, [ax.bbox.width, ax.bbox.height], rtol=2e-14)
        expected_scale = min(ax.bbox.width/4, ax.bbox.height/2)
        np.testing.assert_allclose(lengths, expected_scale, rtol=2e-14)
        np.testing.assert_allclose(corners.mean(axis=0), ax.bbox.get_points().mean(axis=0), rtol=2e-14)
        original = ax.get_position(original=True)
        np.testing.assert_allclose(ax.bbox.width, original.width*fig.bbox.width)
        np.testing.assert_allclose(ax.bbox.height, original.height*fig.bbox.height)


def test_equal_units_derive_three_dimensional_box_from_data():
    fig, ax = p.surf(Field(lambda x, y: x+y), n_pts=7)
    limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    p.matlab_axis_equal(ax)
    spans = np.diff(np.asarray(limits), axis=1).ravel()
    ratios = np.asarray(ax.get_box_aspect()) / spans
    np.testing.assert_allclose(ratios, ratios[0])
    np.testing.assert_array_equal([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()], limits)
    fig.canvas.draw()


def test_uniform_alpha_and_explicit_surface_options():
    f = Field(lambda x, y: x+y)
    _, ax = p.surf(f, n_pts=5, alpha=.5)
    assert isinstance(ax.collections[0], InterpolatedSurface)
    assert ax.collections[0].get_alpha() == .5
    _, ax2 = p.surf(f, n_pts=5, linewidth=.25)
    assert not isinstance(ax2.collections[0], InterpolatedSurface)


def test_colorbar_and_later_clim_updates_use_vertex_cdata():
    fig, ax = p.surf(Field(lambda x, y: x+y), n_pts=5)
    artist = ax.collections[0]
    assert artist.get_array().shape == (25,)
    assert artist.get_cmap() is p.PARULA
    assert fig.colorbar(artist, ax=ax).mappable is artist
    artist.set_clim(-10, 10)
    fig.canvas.draw()
    cdata = artist._source_vertices[..., 2]
    np.testing.assert_array_equal(artist._source_rgba, artist.to_rgba(cdata))


@pytest.mark.parametrize("options", [{"vmin": -10, "vmax": 10}, {"norm": matplotlib.colors.Normalize(-10, 10)}])
def test_explicit_normalization_controls_vertex_colors(options):
    fig, ax = p.surf(Field(lambda x, y: x+y), n_pts=5, **options)
    artist = ax.collections[0]
    assert isinstance(artist, InterpolatedSurface)
    np.testing.assert_array_equal(artist.get_clim(), [-10, 10])
    fig.canvas.draw()
    values = artist._source_vertices[..., 2]
    np.testing.assert_array_equal(artist._source_rgba, p.PARULA(matplotlib.colors.Normalize(-10, 10)(values)))


def test_explicit_facecolors_remain_supported():
    colors = np.zeros((5, 5, 4))
    colors[..., 0] = 1
    colors[..., 3] = 1
    fig, ax = p.surf(Field(lambda x, y: x+y), n_pts=5, facecolors=colors)
    fig.canvas.draw()
    np.testing.assert_array_equal(ax.collections[0].get_facecolors(), np.tile([1, 0, 0, 1], (16, 1)))


def test_supplied_axes_and_two_dimensional_equal_adapter():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    result = p.surf(Field(lambda x, y: x+y), ax=ax, n_pts=5)
    assert result == (fig, ax)
    p.matlab_view(ax, 0, 90)
    p.matlab_axis_equal(ax)
    fig.canvas.draw()
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 2], [0, 1])
    assert p.matlab_axis_equal(ax2) is ax2


@pytest.mark.parametrize("stride", [1, 2])
def test_explicit_edge_fallback_retains_colorbar_and_norm(stride):
    fig, ax = p.surf(Field(lambda x, y: x+y), n_pts=5,
                     edgecolor="black", rstride=stride, cstride=stride,
                     vmin=-10, vmax=10)
    artist = ax.collections[0]
    assert artist.get_array().size == (4//stride)**2
    assert artist.get_cmap() is p.PARULA
    np.testing.assert_array_equal(artist.get_clim(), [-10, 10])
    assert fig.colorbar(artist, ax=ax).mappable is artist
    artist.set_clim(-20, 20)
    fig.canvas.draw()
    np.testing.assert_array_equal(artist.get_clim(), [-20, 20])
