"""Source controls for the Chebop phase-field renderer candidate."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from chebfunjax._quiver_geometry import quiver_line_geometry
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style


def test_quiver_scale_uses_native_euclidean_grid_spacing():
    chebfun_style()
    n = Chebop(lambda t, u: u.diff(2) + u.diff() + u, domain=(0, 1))
    fig, ax = plt.subplots()
    try:
        n.quiver([-2, 2, -3, 3], ax=ax, xpts=5, ypts=3, scale=0.4)
        shaft = ax.collections[0]
        x, y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-3, 3, 3))
        u, v = y, -x-y
        # Independent literal of MATLAB quiver's scale fit: grid spacing is
        # the axis span divided by the number of samples, then the longest
        # vector is fitted to the Euclidean grid spacing.
        factor = 0.4 * np.hypot(4 / 5, 6 / 3) / np.max(np.hypot(u, v))
        got = np.stack(shaft.get_segments())
        np.testing.assert_allclose(got[:, 1] - got[:, 0],
                                   np.stack((u*factor, v*factor), axis=-1).reshape(-1, 2),
                                   rtol=1e-13, atol=1e-13)
    finally:
        plt.close(fig)


def test_quiver_advances_axes_color_cycle_and_explicit_color_does_not():
    chebfun_style()
    n = Chebop(lambda t, u: u.diff(2) + u, domain=(0, 1))
    fig, ax = plt.subplots()
    try:
        n.quiver([-1, 1, -1, 1], ax=ax, xpts=3, ypts=3, color="magenta")
        explicit = ax.collections[-2]
        n.quiver([-1, 1, -1, 1], ax=ax, xpts=3, ypts=3)
        default = ax.collections[-2]
        np.testing.assert_allclose(explicit.get_colors()[0], to_rgba("magenta"))
        np.testing.assert_allclose(default.get_colors()[0], to_rgba("#0072BD"))
        assert ax._get_lines.get_next_color() == "#D95319"
    finally:
        plt.close(fig)


def test_normalized_zero_vectors_are_omitted_without_poisoning_scale():
    chebfun_style()
    n = Chebop(lambda t, u: u.diff(2), domain=(0, 1))
    fig, ax = plt.subplots()
    try:
        n.quiver([-1, 1, -1, 1], ax=ax, xpts=3, ypts=3,
                 normalize=True, scale=0.5)
        got = np.stack(ax.collections[0].get_segments())
        # The middle grid row is the zero field. Normalization creates NaNs
        # there, so the renderer omits those three arrows instead of inventing
        # zero-length geometry.
        assert got.shape == (6, 2, 2)
        assert np.isfinite(got).all()
    finally:
        plt.close(fig)


def test_scale_zero_preserves_raw_shaft_vectors():
    chebfun_style()
    n = Chebop(lambda t, u: u.diff(2) + u.diff() + u, domain=(0, 1))
    fig, ax = plt.subplots()
    try:
        n.quiver([-1, 1, -2, 2], ax=ax, xpts=3, ypts=3, scale=0)
        got = np.stack(ax.collections[0].get_segments())
        x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-2, 2, 3))
        np.testing.assert_allclose(got[:, 1] - got[:, 0],
                                   np.stack((y, -x-y), axis=-1).reshape(-1, 2),
                                   atol=0, rtol=0)
    finally:
        plt.close(fig)


def test_open_arrowhead_matches_source_coordinate_formula():
    # Literal R2025b createLinesForQuiverStruct formula for one arrow.
    # For origin (0, 0) and vector (1, 2), the expanded span is 2 and the
    # source head cutoff is .2 * 2. Keep expected arithmetic independent of
    # the implementation helper's intermediate arrays.
    shafts, heads, bases = quiver_line_geometry(
        np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([2.0]))
    eps = np.finfo(float).eps
    norm = np.sqrt(5.0)
    beta = 0.25 * norm / (norm + eps)
    alpha = 0.33 * (0.2 * 2.0) / norm
    expected_left = np.array([1.0 - alpha * (1.0 + beta * (2.0 + eps)),
                              2.0 - alpha * (2.0 - beta * (1.0 + eps))])
    expected_right = np.array([1.0 - alpha * (1.0 - beta * (2.0 + eps)),
                               2.0 - alpha * (2.0 + beta * (1.0 + eps))])
    np.testing.assert_array_equal(np.asarray(bases), [[0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(shafts), [[[0.0, 0.0], [1.0, 2.0]]],
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(heads),
                               [[expected_left, [1.0, 2.0], expected_right]],
                               rtol=0, atol=0)
