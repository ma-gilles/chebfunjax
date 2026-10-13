"""Source complex plot: angle(-f), flat lower-left colors, fixed HSV scale."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax import plotting as p
from chebfunjax._hsv_colormap import hsv_colormap
from chebfunjax._sphere_surface import InterpolatedSurface


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class ComplexField:
    domain = (-2., 2., -1., 1.)

    def isreal(self):
        return False

    def __call__(self, x, y):
        return x + 1j*y


def test_default_phase_grid_is_source_500(monkeypatch):
    class Captured(Exception):
        pass

    def observe(f, x, y):
        assert x.shape == y.shape == (500, 500)
        np.testing.assert_array_equal(x[0, [0, -1]], [-2, 2])
        np.testing.assert_array_equal(y[[0, -1], 0], [-1, 1])
        raise Captured

    monkeypatch.setattr(p, "_eval_2d_vectorized", observe)
    with pytest.raises(Captured):
        p.plot_chebfun2(ComplexField())


def test_phase_faces_use_fixed_hsv_scale_and_lower_left_vertices():
    fig, ax = p.plot_chebfun2(ComplexField(), n_pts=4)
    artist = ax.collections[0]
    assert not isinstance(artist, InterpolatedSurface)
    assert artist.get_cmap().N == 600
    np.testing.assert_array_equal(artist.get_clim(), [-np.pi, np.pi])
    xx, yy = artist._source_phase_grid
    angles = np.angle(-(xx+1j*yy))
    np.testing.assert_allclose(artist._source_phase_cdata, angles, atol=5e-16)
    expected = hsv_colormap(600)(artist.norm(angles[:-1, :-1])).reshape(-1, 4)
    fig.canvas.draw()
    np.testing.assert_array_equal(artist._facecolor3d, expected)
    assert artist.get_edgecolors().size == 0
    assert not ax.axison
    assert ax.elev == 90 and ax.azim == -90


@pytest.mark.parametrize("entrypoint", ["method", "public"])
def test_public_complex_and_real_plot_dispatch(entrypoint):
    f = cj.chebfun2(lambda x, y: x+1j*y, domain=(-2., 2., -1., 1.))
    plot = (lambda value: value.plot(n_pts=5)) if entrypoint == "method" else (lambda value: cj.plot(value, n_pts=5))
    fig, ax = plot(f)
    assert hasattr(ax.collections[0], "_source_phase_cdata")
    fig.canvas.draw()
    _, real_axes = plot(f.real())
    assert isinstance(real_axes.collections[0], InterpolatedSurface)
