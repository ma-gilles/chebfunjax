"""Public real Trigtech plot data from MATLAB Chebfun 7574c77.

Source @trigtech/plotData.m (one argument), @bndfun/plotData.m mapping,
and @chebfun/plotData.m leading NaN/representation points.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.plotting import _function_lims, matlab_plot, trig_plot_data
from chebfunjax.tech.trigtech import trigpts


@pytest.mark.parametrize("length", [9, 51])
@pytest.mark.parametrize("style", ["-", "-o", "o"])
def test_real_periodic_public_coordinates(length, style):
    nodes = trigpts(length)
    values = jnp.sin(4 * jnp.pi * nodes) + .3 * jnp.cos(2 * jnp.pi * nodes)
    f = cj.chebfun(values, domain=[2, 4], trig=True)
    expected = trig_plot_data(f)
    fig, ax = plt.subplots()
    try:
        matlab_plot(f, style, ax=ax, numpts=17)
        curves = [line for line in ax.lines if line.get_linestyle() != "None"]
        markers = [line for line in ax.lines if line.get_marker() != "None"]
        assert len(curves) == int(style != "o")
        assert len(markers) == int("o" in style)
        for line in curves:
            np.testing.assert_array_equal(line.get_xdata(), expected["xLine"])
            np.testing.assert_array_equal(line.get_ydata(), expected["yLine"])
            assert line.get_xdata()[-1] < 4
        for line in markers:
            np.testing.assert_array_equal(line.get_xdata()[1:], 4 * (nodes + 1) / 2 + 2 * (1 - nodes) / 2)
            np.testing.assert_array_equal(line.get_ydata()[1:], f.funs[0].tech.values)
            assert np.isnan(line.get_xdata()[0])
        np.testing.assert_array_equal(ax.get_xlim(), [2, 4])
        xlim, ylim, default = _function_lims(f)
        np.testing.assert_array_equal(xlim, [2, 4])
        np.testing.assert_array_equal(ylim, [jnp.min(expected["yLine"][1:]), jnp.max(expected["yLine"][1:])])
        assert default
    finally:
        plt.close(fig)


@pytest.mark.parametrize("length", [9, 51])
def test_periodic_array_columns_keep_representation_markers(length):
    nodes = trigpts(length)
    values = jnp.stack([jnp.sin(2 * jnp.pi * nodes), jnp.cos(4 * jnp.pi * nodes)], axis=1)
    f = cj.chebfun(values, domain=[2, 4], trig=True)
    fig, ax = matlab_plot(f, "-o")
    try:
        lines = [line for line in ax.lines if line.get_linestyle() != "None"]
        points = [line for line in ax.lines if line.get_marker() != "None"]
        assert len(lines) == len(points) == 2
        count = 501 if length == 9 else 641
        for column, line in enumerate(lines):
            assert len(line.get_xdata()) == count + 1
            x = line.get_xdata()[1:]
            reference = np.sin(2 * np.pi * (x - 3)) if column == 0 else np.cos(4 * np.pi * (x - 3))
            np.testing.assert_allclose(line.get_ydata()[1:], reference, rtol=0, atol=128 * np.finfo(float).eps)
        for column, point in enumerate(points):
            np.testing.assert_array_equal(point.get_ydata()[1:], values[:, column])
            np.testing.assert_array_equal(point.get_xdata()[1:], 4 * (nodes + 1) / 2 + 2 * (1 - nodes) / 2)
    finally:
        plt.close(fig)


def test_public_periodic_factory_maxlength():
    nodes = trigpts(5216)
    f = cj.chebfun(jnp.cos(100 * jnp.pi * nodes), trig=True)
    fig, ax = matlab_plot(f)
    try:
        assert len(ax.lines) == 1
        x = ax.lines[0].get_xdata()
        assert len(x) == 65537
        assert x[1] == -1 and x[-1] < 1
        np.testing.assert_allclose(ax.lines[0].get_ydata()[1:], np.cos(100 * np.pi * x[1:]), rtol=0, atol=2048 * np.finfo(float).eps)
    finally:
        plt.close(fig)
