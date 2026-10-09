"""Exact visibility and bounded compiler-adapter controls; no pixel fitting."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: independent renderer-boundary assertions.
import pytest

import chebfunjax.plotting as plotting

_SPEC = importlib.util.spec_from_file_location(
    "frozen_visibility", Path(__file__).parent / "fixtures/sphere_visibility_eager.py")
_EAGER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EAGER)
ORTHO = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                  [-1., 0., 0., 0.], [0., 0., 0., 1.]])
PERSP = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                  [0., 0., 0., -1.], [-1., 0., 0., 3.]])
LENGTHS = (0, 1, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257,
           1023, 1024, 1025, 4095, 4096, 4097)


def assert_display(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    assert actual[finite].tobytes() == expected[finite].tobytes()


def test_buckets_order_padding_and_empty(monkeypatch):
    calls = []

    def copying_kernel(points, projection, radius):
        calls.append(points.copy())
        return points

    monkeypatch.setattr(plotting, "_sphere_mask_block", copying_kernel)
    for length in LENGTHS:
        points = np.arange(length * 3, dtype=float).reshape(length, 3)
        if length:
            points[0] = [0., -0., np.nan]
        calls.clear()
        actual = plotting._sphere_display_vertices(points, ORTHO, 1.)
        assert actual.tobytes() == points.tobytes()
        expected_counts = [min(4096, length - start) for start in range(0, length, 4096)]
        assert len(calls) == len(expected_counts)
        for block, count in zip(calls, expected_counts):
            assert len(block) == max(32, 1 << (count - 1).bit_length())
            assert np.isnan(block[count:]).all()


def test_empty_does_not_evaluate_projection():
    actual = plotting._sphere_display_vertices(np.empty((0, 3)), None, None)
    assert actual.shape == (0, 3)


def test_all_eight_compiled_shapes_match_frozen_eager():
    seed = np.array([[2., -0., 0.], [-2., 0., -0.], [0., 2., -0.],
                     [np.nan, np.nan, np.nan], [.5, 0., 0.]])
    visible = np.asarray(_EAGER._sphere_visible_vertices(seed, ORTHO, 1.))
    expected_seed = np.where(visible[:, None], seed, np.nan)
    for count in (32, 64, 128, 256, 512, 1024, 2048, 4096, 4097):
        indices = np.arange(count) % len(seed)
        actual = plotting._sphere_display_vertices(seed[indices], ORTHO, 1.)
        assert_display(actual, expected_seed[indices])
    assert plotting._sphere_mask_block._cache_size() <= 8


@pytest.mark.parametrize("projection", [ORTHO, PERSP], ids=["ortho", "perspective"])
def test_boundary_nan_camera_and_signed_zero_match_eager(projection):
    points = np.array([[2., -0., 0.], [.5, 0., 0.], [1., 0., -0.],
                       [-1., 0., 0.], [0., 2., 0.], [3., 0., 0.],
                       [np.nextafter(1., 0.), 0., 0.],
                       [np.nextafter(1., np.inf), 0., 0.],
                       [np.nan, 0., 0.], [0., np.inf, 0.]])
    for radius in (1., .99):
        visible = _EAGER._sphere_visible_vertices(points, projection, radius)
        expected = np.asarray(jnp.where(visible[:, None], points, jnp.nan))
        assert_display(plotting._sphere_display_vertices(points, projection, radius), expected)


@pytest.mark.parametrize("kind", ["ortho", "persp"])
def test_camera_roll_scale_and_radius_are_dynamic(kind):
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-2., 2.), ylim=(-3., 3.), zlim=(-1., 1.))
        ax.set_box_aspect((2., 3., 1.))
        ax.set_proj_type(kind)
        points = np.array([[2., 0., 0.], [-2., 0., 0.], [0., 2., 0.],
                           [0., -2., 0.], [.5, .5, 0.], [0., 0., 2.]])
        outputs = []
        for azimuth, elevation, roll, radius in ((0., 0., 0., 1.),
                                                 (137., 23., 31., .99)):
            ax.view_init(elev=elevation, azim=azimuth, roll=roll)
            projection = ax.get_proj()
            visible = _EAGER._sphere_visible_vertices(points, projection, radius)
            expected = np.asarray(jnp.where(visible[:, None], points, jnp.nan))
            actual = plotting._sphere_display_vertices(points, projection, radius)
            assert_display(actual, expected)
            outputs.append(np.isnan(actual))
        assert not np.array_equal(*outputs)
    finally:
        plt.close(fig)
