"""Source contracts: @chebfun/arrowplot.m and @chebtech/plotData.m, 7574c77."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax import chebfun
from chebfunjax.plotting import arrowplot


@pytest.fixture(autouse=True)
def close():
    yield
    plt.close("all")

def test_native_complex_curve_chebyshev_grid():
    t = chebfun(lambda x: x)
    _, ax = arrowplot(t, t**2)
    # Source plotData oversamples this degree-two polynomial at 501
    # second-kind nodes; this independently specifies the curve data.
    # @chebfun/plotData prepends a NaN row even for one piece.
    x = -np.cos(np.pi * np.arange(501) / 500)
    line = ax.lines[0]
    assert len(line.get_xdata()) == 502
    assert np.isnan(line.get_xdata()[0]) and np.isnan(line.get_ydata()[0])
    np.testing.assert_allclose(line.get_xdata()[1:], x, rtol=0, atol=2e-14)
    np.testing.assert_allclose(line.get_ydata()[1:], x*x, rtol=0, atol=2e-14)

def test_native_held_color_follows_plotted_line():
    t = chebfun(lambda x: x)
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=["magenta", "cyan"])
    ax.plot([0, 1], [0, 1])
    arrowplot(t, t**2, ax=ax)
    assert ax.lines[-1].get_color() == "cyan"
    assert ax.texts[-1].arrow_patch.get_edgecolor() == matplotlib.colors.to_rgba("cyan")

def test_nonzero_stationary_endpoint_keeps_annotation():
    f = chebfun(lambda x: 2 + 0*x)
    g = chebfun(lambda x: 3 + 0*x)
    _, ax = arrowplot(f, g)
    assert len(ax.texts) == 1
    np.testing.assert_array_equal(ax.texts[0].xy, [2, 3])
    np.testing.assert_array_equal(ax.texts[0].xyann, [2, 3])

def test_real_single_argument_requires_second_component():
    with pytest.raises(ValueError, match="two real"):
        arrowplot(chebfun(lambda x: x))


def test_real_parametric_curve_stays_in_phase_plane():
    t = chebfun(lambda x: x)
    _, ax = arrowplot(t, 0*t)
    x = -np.cos(np.pi * np.arange(501) / 500)
    line = ax.lines[0]
    assert len(line.get_xdata()) == 502
    np.testing.assert_allclose(line.get_xdata()[1:], x, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(line.get_ydata()[1:], np.zeros(501))


def test_markers_use_representation_nodes():
    t = chebfun(lambda x: x)
    _, ax = arrowplot(t, t**2, marker="o")
    marker_line = next(line for line in ax.lines if line.get_marker() == "o")
    x = np.asarray(marker_line.get_xdata())
    y = np.asarray(marker_line.get_ydata())
    assert len(x) == 4 and np.isnan(x[0])
    np.testing.assert_allclose(x[1:], [-1, 0, 1], rtol=0, atol=2e-14)
    np.testing.assert_allclose(y[1:], [1, 0, 1], rtol=0, atol=2e-14)
