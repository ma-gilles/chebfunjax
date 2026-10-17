"""Native numeric preflush and readable compatibility geometry observations."""

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from chebfunjax import quiver_points
from chebfunjax._quiver_geometry import quiver_line_geometry
from chebfunjax.plotting import _quiver_source_scale

FIXTURES = Path(__file__).parents[2] / "fixtures/quiver_points_preflush_2025b"


def decode(a):
    return np.array(
        [np.frombuffer(bytes.fromhex(h), dtype=">f8")[0] for h in a["real_hex"]]
    ).reshape(a["size"], order="F")


@pytest.mark.parametrize("case", range(1, 9))
def test_native_scale_and_compatibility_lines(case):
    r = json.loads((FIXTURES / f"case{case}.json").read_text())
    x, y, u, v = (decode(r[k]) for k in ("x", "y", "u", "v"))
    factor = float(decode(r["factor"]).item())
    expected = float(decode(r["properties"]["ScaleFactor"]).item())
    scale = _quiver_source_scale(x, y, u, v, factor)
    assert abs(float(scale) - expected) <= abs(np.spacing(expected))
    shafts, heads, _ = quiver_line_geometry(x, y, u * scale, v * scale)
    for actual, line, n in [
        (shafts, r["compatibility_lines"][0], 3),
        (heads, r["compatibility_lines"][1], 4),
    ]:
        target = np.stack((decode(line["XData"]).ravel(), decode(line["YData"]).ravel()), axis=-1)
        target = target.reshape(-1, n, 2)[:, : n - 1, :]
        np.testing.assert_allclose(
            actual, target, rtol=0, atol=4 * np.finfo(float).eps, equal_nan=True
        )
    fig, ax = plt.subplots()
    try:
        ax.set_xlim(-3, 4)
        ax.set_ylim(-2, 5)
        ax.set_aspect("equal")
        returned = quiver_points(x, y, u, v, ax=ax, autoscale_factor=factor, color="k", linewidth=2)
        assert returned == (fig, ax)
        assert ax.get_xlim() == (-3.0, 4.0) and ax.get_ylim() == (-2.0, 5.0)
        assert not ax.get_autoscalex_on() and not ax.get_autoscaley_on()
        assert ax.get_aspect() == 1.0
        assert len(ax.collections) == 2
        assert ax.collections[0].get_linewidths()[0] == 2
        np.testing.assert_array_equal(ax.collections[0].get_colors()[0], [0, 0, 0, 1])
        np.testing.assert_allclose(ax.collections[0].get_segments(), shafts, rtol=0, atol=0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1), (1, 1)])
def test_shapes_and_options(shape):
    a = np.arange(np.prod(shape)).reshape(shape)
    fig, ax = quiver_points(a, a, a + 1, a + 2, show_arrow_head=False, colors="r", lw=3)
    try:
        assert len(ax.collections[0].get_segments()) == a.size
        assert not ax.collections[1].get_visible()
        assert ax.get_autoscalex_on() and ax.get_autoscaley_on()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("a", [[], 1, [[[1]]]])
def test_invalid_shape(a):
    with pytest.raises(ValueError):
        quiver_points(a, a, a, a)


@pytest.mark.parametrize("a", [[complex(1, 1)], [float("nan")], [float("inf")]])
def test_nonfinite_or_complex(a):
    with pytest.raises(NotImplementedError):
        quiver_points(a, a, a, a)


def test_invalid_options_and_mismatch():
    with pytest.raises(ValueError):
        quiver_points([1], [1], [1, 2], [1])
    with pytest.raises(ValueError):
        quiver_points([1], [1], [1], [1], scale=1)
    with pytest.raises(ValueError):
        quiver_points([1], [1], [1], [1], autoscale_factor=-1)
