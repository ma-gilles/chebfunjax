"""ColorOrderIndex controls for the pinned Chebfun plotting adapter."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import matlab_plot, plotregion

COLORS = ["purple", "orange", "green", "cyan"]


def _axis():
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=COLORS)
    return fig, ax


def _region_color_after(ax, *args, **kwargs):
    f = cj.chebfun([1.0, 0.2, -0.05], domain=(-1.0, 1.0), coeffs=True)
    matlab_plot(*args, ax=ax, **kwargs)
    plotregion(f, ax=ax)
    return ax.lines[-2].get_color()


def test_explicit_color_keyword_consumes_one_cycle_entry_but_linespec_does_not():
    fig, ax = _axis()
    assert _region_color_after(ax, cj.chebfun(lambda x: x), color="black") == "orange"
    plt.close(fig)

    fig, ax = _axis()
    assert _region_color_after(ax, cj.chebfun(lambda x: x), "b") == "purple"
    plt.close(fig)


def test_default_color_and_marker_points_consume_only_the_source_line_entry():
    fig, ax = _axis()
    f = cj.chebfun(lambda x: x)
    matlab_plot(f, ax=ax)
    plotregion(f, ax=ax)
    assert ax.lines[-2].get_color() == "orange"
    plt.close(fig)

    fig, ax = _axis()
    matlab_plot(f, ax=ax, color="black", marker="o")
    assert sum(line.get_marker() == "o" for line in ax.lines) == 1
    plotregion(f, ax=ax)
    assert ax.lines[-2].get_color() == "orange"
    plt.close(fig)


def test_multicolumn_and_piecewise_routes_advance_per_source_line_artist():
    fig, ax = _axis()
    x = np.linspace(-1.0, 1.0, 9)
    y = np.column_stack((x, x * x))
    matlab_plot(x, y, ax=ax, color="black")
    assert len(ax.lines) == 2
    plotregion(cj.chebfun(lambda t: t), ax=ax)
    assert ax.lines[-2].get_color() == "green"
    plt.close(fig)

    fig, ax = _axis()
    piecewise = cj.chebfun(lambda t: t, domain=(-1.0, 0.0, 1.0))
    matlab_plot(piecewise, ax=ax, color="black", marker="o")
    source_lines = [line for line in ax.lines if line.get_marker() == "None"]
    assert len(source_lines) == 1
    plotregion(cj.chebfun(lambda t: t), ax=ax)
    assert ax.lines[-2].get_color() == "orange"
    plt.close(fig)


def test_numeric_x_y_property_color_and_colored_linespec_keep_native_split():
    x = np.linspace(-1.0, 1.0, 7)
    y = x * x

    fig, ax = _axis()
    matlab_plot(x, y, "o", ax=ax, markersize=4, color="b",
                markerfacecolor="b")
    assert ax.lines[0].get_color() == "b"
    assert ax.lines[0].get_marker() == "o"
    plotregion(cj.chebfun(lambda t: t), ax=ax)
    assert ax.lines[-2].get_color() == "orange"
    plt.close(fig)


def test_colored_linespec_suppresses_cycle_even_when_color_property_overrides():
    fig, ax = _axis()
    f = cj.chebfun(lambda t: t)
    matlab_plot(f, "b", ax=ax, color="black")
    assert ax.lines[0].get_color() == "black"
    plotregion(f, ax=ax)
    assert ax.lines[-2].get_color() == "purple"
    plt.close(fig)

    fig, ax = _axis()
    x = np.linspace(-1.0, 1.0, 7)
    y = x * x
    matlab_plot(x, y, "bo", ax=ax, color="black", markersize=4)
    assert ax.lines[0].get_color() == "black"
    assert ax.lines[0].get_marker() == "o"
    plotregion(cj.chebfun(lambda t: t), ax=ax)
    assert ax.lines[-2].get_color() == "purple"
    plt.close(fig)

    fig, ax = _axis()
    matlab_plot(x, y, "bo", ax=ax, markersize=4, markerfacecolor="b")
    assert ax.lines[0].get_color() == "b"
    plotregion(cj.chebfun(lambda t: t), ax=ax)
    assert ax.lines[-2].get_color() == "purple"
    plt.close(fig)
