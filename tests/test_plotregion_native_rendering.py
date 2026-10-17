"""Artist and axes-state controls for native plotregion rendering."""

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.plotting import plotregion


def test_default_chebtech_draws_boundary_and_black_domain_without_decoration():
    f = cj.chebfun([1.0, 0.2, -0.05], domain=(-2.0, 3.0), coeffs=True)
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["purple", "orange"])
    ax.set_title("existing title")
    ax.set_xlabel("existing x")
    ax.set_ylabel("existing y")
    ax.set_aspect(0.75)
    ax.grid(True)
    grid_visible = ax.get_xgridlines()[0].get_visible()
    # Compare the caller's layout rectangle, not the active box Matplotlib
    # derives after the new source limits interact with the preserved aspect.
    axes_position = tuple(ax.get_position(original=True).bounds)
    adjustable = ax.get_adjustable()
    anchor = ax.get_anchor()
    layout_engine = fig.get_layout_engine()
    figure_facecolor = fig.get_facecolor()

    plotregion(f, ax=ax)

    assert len(ax.lines) == 2  # ellipse and native black '+-' domain line
    assert len(ax.collections) == 0  # no implicit fill_between artist
    assert ax.lines[0].get_color() == "purple"  # current axes cycle
    assert ax.lines[1].get_color() == "k"
    assert ax.lines[1].get_marker() == "+"
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [-2.0, 3.0])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0.0, 0.0])
    assert ax.get_title() == "existing title"
    assert ax.get_xlabel() == "existing x"
    assert ax.get_ylabel() == "existing y"
    assert ax.get_aspect() == 0.75
    assert ax.get_xgridlines()[0].get_visible() == grid_visible
    np.testing.assert_array_equal(ax.get_position(original=True).bounds, axes_position)
    assert ax.get_adjustable() == adjustable
    assert ax.get_anchor() == anchor
    assert fig.get_layout_engine() is layout_engine
    assert fig.get_facecolor() == figure_facecolor
    plt.close(fig)


def test_piecewise_domain_line_uses_all_native_domain_breakpoints():
    pieces = [
        _Piece.from_coeffs(jnp.asarray([1.0, 0.2, 0.1]), -3.0, -0.25),
        _Piece.from_coeffs(jnp.asarray([2.0, -0.3, 0.12, 0.04]), -0.25, 2.0),
    ]
    f = Chebfun(funs=pieces, domain=Domain((-3.0, -0.25, 2.0)))
    fig, ax = plotregion(f)

    assert len(ax.lines) == 3
    np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [-3.0, -0.25, 2.0])
    np.testing.assert_array_equal(ax.lines[-1].get_ydata(), [0.0, 0.0, 0.0])
    assert ax.lines[-1].get_color() == "k"
    assert ax.lines[-1].get_marker() == "+"
    plt.close(fig)


def test_color_cycle_continues_across_calls_and_linewidth_keyword_is_safe():
    first = cj.chebfun([1.0, 0.2, -0.05], domain=(-1.0, 1.0), coeffs=True)
    second = cj.chebfun([0.5, -0.1, 0.03], domain=(2.0, 5.0), coeffs=True)
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["magenta", "cyan", "green"])

    plotregion(first, ax=ax, linewidth=2.75)
    plotregion(second, ax=ax, linewidth=3.25)

    assert [ax.lines[0].get_color(), ax.lines[2].get_color()] == ["magenta", "cyan"]
    assert ax.lines[0].get_linewidth() == 2.75
    assert ax.lines[2].get_linewidth() == 3.25
    assert ax.lines[1].get_color() == ax.lines[3].get_color() == "k"
    plt.close(fig)


def test_explicit_title_and_decorative_adapter_are_opt_in():
    f = cj.chebfun([1.0, 0.2, -0.05], domain=(-1.0, 1.0), coeffs=True)
    fig, ax = plotregion(f, title="caller title", color="red", linewidth=2.0)
    assert ax.get_title() == "caller title"
    assert ax.lines[0].get_color() == "red"
    assert ax.lines[0].get_linewidth() == 2.0
    assert len(ax.collections) == 0
    plt.close(fig)

    fig, ax = plotregion(f, decorate=True)
    assert ax.get_title() == "Region of analyticity"
    assert ax.get_xlabel() == "Re(z)"
    assert ax.get_ylabel() == "Im(z)"
    assert ax.get_aspect() == 1.0
    assert len(ax.collections) == 1  # legacy translucent ellipse fill
    assert len(ax.lines) == 3  # ellipse, domain line, legacy gray real axis
    plt.close(fig)


def test_trig_boundary_and_auxiliary_keep_source_coordinates_and_color():
    f = cj.chebfun(jnp.arange(1.0, 6.0), domain=[2.0, 8.0], trig=True)
    fig, ax = plotregion(f, eps=1e-12)

    assert len(ax.lines) == 3
    height = 3.0 * np.log(1.0e12) / (np.pi * 2.0)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [2.0, 8.0, np.nan, 2.0, 8.0])
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), [height, height, np.nan, -height, -height]
    )
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [2.0, 2.0, np.nan, 8.0, 8.0])
    assert ax.lines[1].get_linestyle() == ":"
    assert ax.lines[1].get_color() == ax.lines[0].get_color()
    np.testing.assert_array_equal(ax.lines[2].get_xdata(), [2.0, 8.0])
    assert ax.lines[2].get_color() == "k"
    assert len(ax.collections) == 0
    plt.close(fig)


def test_three_body_uses_public_plotregion_for_both_outputs():
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/ode-nonlin/three_body_problem.py"
    ).read_text()
    tree = ast.parse(source)
    imports_plotregion = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "chebfunjax.plotting"
        and any(alias.name == "plotregion" for alias in node.names)
        for node in ast.walk(tree)
    )
    calls_plotregion = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plotregion"
    ]
    local_copy = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_plotregion"
        for node in tree.body
    )
    assert imports_plotregion
    assert len(calls_plotregion) == 2
    assert not local_copy
