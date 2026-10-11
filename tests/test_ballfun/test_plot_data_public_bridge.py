"""Public Ballfun source-data plumbing; existing lighting is not qualified."""

import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: Matplotlib host artist/data assertions.
import pytest

import chebfunjax._ball_plot_data as data_module
import chebfunjax._sphere_surface as sphere_surface
import chebfunjax.plotting as plotting
from chebfunjax.ballfun.ballfun import Ballfun


def polynomial_ball():
    # .25 + .125*z = .25 + .125*r*cos(theta), from source CFF modes.
    coefficients = jnp.zeros((2, 1, 3), dtype=jnp.complex128)
    coefficients = coefficients.at[0, 0, 1].set(.25)
    coefficients = coefficients.at[1, 0, 0].set(.0625)
    coefficients = coefficients.at[1, 0, 2].set(.0625)
    return Ballfun.from_coeffs(coefficients, is_real=True)


@pytest.mark.parametrize("route", ["plot", "surf", "dispatch"])
def test_public_default_source_data_and_real_five_surface_artist(monkeypatch, route, tmp_path):
    ball = polynomial_ball()
    original_data = data_module.ball_plot_data
    source_records, surfaces, normalizer_inputs = [], [], []

    def source_spy(value):
        assert value is ball
        result = original_data(value)
        source_records.append(result)
        return result

    def forbidden_fevalm(*args, **kwargs):
        raise AssertionError("default source data must use coefficient transforms")

    original_group = sphere_surface.InterpolatedSurfaceGroup

    def group_spy(items):
        surfaces.append(tuple(tuple(np.asarray(v).copy() for v in item)
                               for item in items))
        return original_group(items)

    original_normalize = plotting._normalize_values

    def normalize_spy(values):
        normalizer_inputs.append(np.asarray(values).copy())
        return original_normalize(values)

    monkeypatch.setattr(data_module, "ball_plot_data", source_spy)
    monkeypatch.setattr(Ballfun, "fevalm", forbidden_fevalm)
    monkeypatch.setattr(sphere_surface, "InterpolatedSurfaceGroup", group_spy)
    monkeypatch.setattr(plotting, "_normalize_values", normalize_spy)
    fig = plt.figure(figsize=(2, 2), dpi=96)
    try:
        ax = fig.add_subplot(111, projection="3d")
        result = (plotting.plot_dispatch(ball, ax=ax) if route == "dispatch"
                  else getattr(ball, route)(ax=ax))
        assert result == (fig, ax)
        assert len(source_records) == 1
        assert len(surfaces) == 1
        assert len(surfaces[0]) == 5
        assert len(ax.collections) == 1
        artist = ax.collections[0]
        assert isinstance(artist, original_group)
        assert len(artist._source_surfaces) == 5
        data = source_records[0]
        for captured, source, (stored_xyz, stored_rgba) in zip(
                surfaces[0], data.surfaces, artist._source_surfaces):
            for actual, expected in zip(captured[:3], (source.x, source.y, source.z)):
                assert actual.tobytes() == np.asarray(expected).tobytes()
            np.testing.assert_array_equal(
                np.stack(captured[:3], axis=-1), stored_xyz)
            np.testing.assert_array_equal(captured[3], stored_rgba)
            assert captured[3].shape == np.asarray(source.values).shape + (4,)
            assert np.all(np.isfinite(captured[3]))
        expected_values = np.concatenate([np.asarray(s.values).ravel() for s in data.surfaces])
        assert len(normalizer_inputs) == 1
        assert normalizer_inputs[0].tobytes() == expected_values.tobytes()
        for limits in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()):
            assert limits == (-1., 1.)
        aspect = ax.get_box_aspect()
        np.testing.assert_array_equal(aspect, np.full(3, aspect[0]))
        fig.canvas.draw()
        assert np.asarray(fig.canvas.buffer_rgba()).shape == (192, 192, 4)
        output = Path(os.environ.get("CHEBFUN_RUNTIME_REPORT", tmp_path)) / "bridge_figures"
        output.mkdir(exist_ok=True)
        fig.savefig(output / f"{route}.png", dpi=96)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("style,count", [("WedgeAz", 3), ("WedgePol", 5)])
def test_unchanged_wedge_setup_and_actual_artist_routes(monkeypatch, style, count):
    ball = polynomial_ball()
    calls = []

    def sample(self, r, lam, theta):
        assert self is ball
        shape = (len(r), len(lam), len(theta))
        calls.append(shape)
        return jnp.full(shape, .25)

    def forbidden_default(value):
        raise AssertionError("wedge must retain its existing fevalm route")

    monkeypatch.setattr(Ballfun, "fevalm", sample)
    monkeypatch.setattr(data_module, "ball_plot_data", forbidden_default)
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection="3d")
        assert ball.plot(style, ax=ax) == (fig, ax)
        assert len(ax.collections) == count
        assert len(calls) == count
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["coefficients", "factory"])
def test_empty_default_rejects_before_creating_axes(monkeypatch, kind):
    empty = (Ballfun.empty() if kind == "factory" else
             Ballfun.from_coeffs(jnp.empty((0, 0, 0)), is_real=True))

    def forbidden_figure(*args, **kwargs):
        raise AssertionError("source empty validation precedes graphics setup")

    monkeypatch.setattr(plt, "figure", forbidden_figure)
    with pytest.raises(ValueError, match="CHEBFUN:BALLFUN:plot:isempty"):
        empty.plot()


def test_complex_public_default_warns_and_passes_only_real_cdata(monkeypatch):
    ball = Ballfun.from_coeffs(jnp.asarray([[[.25 + .5j]]]), is_real=False)
    original = plotting._normalize_values
    captured = []

    def normalize(values):
        captured.append(np.asarray(values).copy())
        return original(values)

    monkeypatch.setattr(plotting, "_normalize_values", normalize)
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection="3d")
        with pytest.warns(UserWarning, match="CHEBFUN:BALLFUN:plot:isReal"):
            ball.plot(ax=ax)
        assert len(captured) == 1
        np.testing.assert_array_equal(captured[0], .25)
        assert len(ax.collections) == 1
        artist = ax.collections[0]
        assert isinstance(artist, sphere_surface.InterpolatedSurfaceGroup)
        assert len(artist._source_surfaces) == 5
    finally:
        plt.close(fig)
