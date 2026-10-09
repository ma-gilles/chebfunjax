"""Pinned contour.m79–83 FFT factor samples and longitude closure.

Independent analytic grids retain the existing512eps bound. High frequencies
exercise actual aliasing, with Horner evaluation forbidden to distinguish
source paths even when both paths represent the same mathematical values.
"""
# uses-numpy: independent analytic grid and Matplotlib capture assertions.
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from chebfunjax import plotting
from chebfunjax.spherefun import _plus
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech import trigtech


@pytest.mark.parametrize("frequency,n_pts", [(1, 11), (9, 8)])
def test_contour_native_samples_grid_and_aliasing(monkeypatch, frequency, n_pts):
    lon = jnp.linspace(-jnp.pi, jnp.pi, 23)[:-1]
    lat = jnp.linspace(0., jnp.pi, 21)
    ll, tt = jnp.meshgrid(lon, lat)
    power = 9 if frequency == 9 else 2
    f = Spherefun.from_values(2+jnp.cos(tt)+.25*jnp.sin(tt)**power*jnp.cos(frequency*ll))
    if frequency == 9:
        # Independently qualify constructor/sample output before plotting dispatch.
        native_lon = np.linspace(-np.pi, np.pi, n_pts)[:-1]
        native_lat = np.linspace(0., np.pi, n_pts)
        native_ll, native_tt = np.meshgrid(native_lon, native_lat)
        analytic = 2+np.cos(native_tt)+.25*np.sin(native_tt)**power*np.cos(frequency*native_ll)
        np.testing.assert_allclose(np.asarray(f.sample(n_pts-1, n_pts)), analytic,
                                   rtol=0, atol=512*np.finfo(float).eps)
    captures, samples, aliases = [], [], []
    original_contour = Axes.contour
    original_sample = Spherefun.sample
    original_alias = _plus._sample

    def capture(self, x, y, z, *args, **kwargs):
        captures.append((np.asarray(x), np.asarray(y), np.asarray(z)))
        return original_contour(self, x, y, z, *args, **kwargs)

    def sample(self, m=None, n=None):
        samples.append((m, n))
        return original_sample(self, m, n)

    def alias(factors, n):
        aliases.append((max(len(f.coeffs) for f in factors), n))
        return original_alias(factors, n)

    def forbidden(*args, **kwargs):
        pytest.fail("Native contour must sample factors, not use direct Horner evaluation")

    monkeypatch.setattr(Axes, "contour", capture)
    monkeypatch.setattr(Spherefun, "sample", sample)
    monkeypatch.setattr(Spherefun, "__call__", forbidden)
    monkeypatch.setattr(trigtech, "_trig_eval_np", forbidden)
    monkeypatch.setattr(_plus, "_sample", alias)
    fig, ax = plotting.contour_sphere(f, n_pts=n_pts, levels=[1.5, 2., 2.5], linewidth=2.)
    try:
        assert samples == [(n_pts-1, n_pts)]
        assert len(captures) == 1
        x, y, values = captures[0]
        np.testing.assert_array_equal(x, np.linspace(-np.pi, np.pi, n_pts))
        np.testing.assert_array_equal(y, np.linspace(0., np.pi, n_pts))
        assert values.shape == (n_pts, n_pts)
        np.testing.assert_array_equal(values[:, -1], values[:, 0])
        longitude, latitude = np.meshgrid(x, y)
        expected = 2+np.cos(latitude)+.25*np.sin(latitude)**power*np.cos(frequency*longitude)
        np.testing.assert_allclose(values, expected, rtol=0, atol=512*np.finfo(float).eps)
        assert all(line.get_linewidth() == 2. for line in ax.lines)
        if frequency == 9:
            assert any(length > n and n == n_pts-1 for length, n in aliases)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_contour_closes_exact_sample_payload_without_transposition(monkeypatch):
    captured = []
    original = Axes.contour
    payload = np.arange(20., dtype=float).reshape(5, 4)

    class SampleOnly:
        def sample(self, m, n):
            assert (m, n) == (4, 5)
            return payload

    def capture(self, x, y, z, *args, **kwargs):
        captured.append(np.asarray(z))
        return original(self, x, y, z, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", capture)
    fig, _ = plotting.contour_sphere(SampleOnly(), n_pts=5, levels=[4., 10., 16.])
    try:
        np.testing.assert_array_equal(captured[0][:, :4], payload)
        np.testing.assert_array_equal(captured[0][:, 4], payload[:, 0])
    finally:
        plt.close(fig)
