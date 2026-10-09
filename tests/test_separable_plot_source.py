"""Source sampling and scalar surf policies from separableApprox (7574c77)."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax import plotting


class SampledFunction:
    domain = (-2.0, 3.0, -4.0, 5.0)

    def __init__(self, callback):
        self.callback = callback
        self.samples = []

    def __call__(self, x, y):
        self.samples.append((np.asarray(x), np.asarray(y)))
        return self.callback(x, y)


@pytest.mark.parametrize('kind', ['surf', 'contour'])
@pytest.mark.parametrize('count', [None, 17])
def test_source_uniform_plot_grid(kind, count):
    f = SampledFunction(lambda x, y: x + 2*y)
    options = {} if count is None else {'n_pts': count}
    fig, _ = getattr(plotting, kind)(f, **options)
    try:
        n = 200 if count is None else count
        x, y = f.samples[0]
        assert x.shape == y.shape == (n, n)
        assert np.array_equal(x[0], np.linspace(-2, 3, n))
        assert np.array_equal(y[:, 0], np.linspace(-4, 5, n))
    finally:
        plt.close(fig)


@pytest.mark.parametrize('amplitude,flatten', [(1e-11, True), (3e-11, False)])
def test_surf_uses_matrix_infinity_norm_for_roundoff_color(monkeypatch, amplitude, flatten):
    f = SampledFunction(lambda x, y: 1 + amplitude*(x+2)/5)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    captured = []
    original = ax.plot_surface

    def record(x, y, z, **options):
        captured.append(np.asarray(z))
        return original(x, y, z, **options)

    monkeypatch.setattr(ax, 'plot_surface', record)
    try:
        plotting.surf(f, ax=ax, n_pts=10)
        if flatten:
            assert np.all(captured[0] == 1)
        else:
            assert np.any(captured[0] != 1)
            x, y = f.samples[0]
            assert np.array_equal(captured[0], np.asarray(f.callback(jnp.asarray(x), jnp.asarray(y))))
    finally:
        plt.close(fig)


@pytest.mark.parametrize('filled', [False, True])
def test_contour_colorbar_preserves_levels(filled):
    f = SampledFunction(lambda x, y: x + y)
    levels = [-1., 0., 1., 2.]
    fig, ax = plotting.contour(f, levels=levels, n_pts=17, filled=filled, colorbar=True)
    try:
        assert len(fig.axes) == 2
        colored = next(c for c in ax.collections if c.colorbar is not None)
        assert np.array_equal(colored.levels, levels)
        assert colored.colorbar.ax is fig.axes[1]
    finally:
        plt.close(fig)
