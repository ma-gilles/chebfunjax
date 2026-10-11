"""Delay differential equations in Chebfun.

Translation of ode-nonlin/DelayDifferentialEquations.m: a tour of
delay, pantograph, state-dependent and integro-differential equations,
first by explicit spectral discretization (chebpts / diffmat / barymat)
and then through chebop, whose operators may evaluate the unknown at
mapped arguments -- x(q*t), x(max(t-p, 0)), even x(x).

Original: https://www.chebfun.org/examples/ode-nonlin/DelayDifferentialEquations.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax.numpy as jnp

from chebfunjax import chebfun, volt
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.diffmat import cumsummat, diffmat
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts as _chebpts_ref


def chebpts_dom(n, dom):
    """2nd-kind points mapped onto [dom[0], dom[1]] (MATLAB chebpts(n, dom))."""
    x = np.asarray(_chebpts_ref(n), dtype=float)
    a, b = float(dom[0]), float(dom[1])
    return a + (b - a) * (x + 1.0) / 2.0

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

SOURCE_FIGURES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 12, 11: 13, 12: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24}
OUTPUT_SLOT = [0]
SAVED_IMAGES = set()


def _save(fig, *, source_axes=False, spy=False):
    mapping = {8: 9, 9: 11} if spy else SOURCE_FIGURES
    image_id = mapping[OUTPUT_SLOT[0]]
    if image_id in SAVED_IMAGES:
        raise RuntimeError(f"Duplicate source figure {image_id}")
    SAVED_IMAGES.add(image_id)
    fig.set_facecolor("white")
    if not source_axes:
        fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"DelayDifferentialEquations_{image_id:02d}.png"),
        **({"layout": "matlab"} if spy else {}))
    if os.environ.get("CHEBFUNJAX_DDE_FIGURE_CAPTURE_DIR"):
        _capture_figure_data(fig, image_id, sys._getframe(1).f_locals)
    plt.close(fig)


def _source_delay_curve(points, values, section):
    """Render the scoped constant-delay curves with the reference axes."""
    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    ax.plot(points, values, lw=1.4)
    xmax, ymax, ystep = (3, 10, 2) if section == 8 else (2, 60, 10)
    ax.set(xlim=(0, xmax), ylim=(0, ymax),
           xticks=np.arange(0, xmax + .25, .5),
           yticks=np.arange(0, ymax + ystep, ystep))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_delay_spy(matrix):
    """Render exact nonzeros with the qualified historical public spy helper."""
    from chebfunjax.plotting import spy

    m, n = matrix.shape
    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    spy(matrix, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, n+1, 20 if n == 66 else 50))
    ax.set_yticks(np.arange(0, m+1, 10 if m == 66 else 20))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_volterra_curve(sol, numerical):
    """Render source slot 11 with separate representation-node markers."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(sol, "-", numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(.86, 1),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(.86, 1.001, .02))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_state_delay_curve(hand_t, hand_x, numerical):
    """Source hold-on overlay using the retained hand-Newton curve."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    ax.plot(hand_t, hand_x, "-")
    time = chebfun(lambda t: t, domain=(0, 1))
    matlab_plot(time, numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(.4, 1),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(.4, 1.001, .1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_pantograph_curve(exact, numerical):
    """Source slot15 Chebfun lines and representation-node markers."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact, "-", numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(.4, 1),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(.4, 1.001, .1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_system_display(label, functions):
    """Format the native column ChebMatrix display for source slot16."""
    print(f"{label} =\n   {len(functions)} x 1 chebmatrix of block types:\n")
    for function in functions:
        print("    {'" + type(function).__name__.lower() + "'}")
    print()


def _source_ddex1de(t, y, Z):
    """Literal installed MATLAB DDEX1 derivative callback."""
    return jnp.array([Z[0, 0], Z[0, 0] + Z[1, 1], y[1]])


def _source_ddex1hist(t):
    """Literal installed MATLAB DDEX1 constant history callback."""
    return jnp.ones(3)


def _source_discrete_delay_system_curve(components, mesh, values):
    """Render the source Chebop/DDESD overlay on its natural mesh."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    for component in components:
        matlab_plot(component, "-b", ax=ax)
    matlab_plot(mesh, np.asarray(values).T, "--r", ax=ax)
    ax.set(xlim=(0, 2), ylim=(0, 12),
           xticks=np.arange(0, 2.01, .5), yticks=np.arange(0, 12.01, 2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_state_dependent_system_curve(exact, numerical):
    """Render the source final state-dependent system with reference axes."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact[0], "-b", exact[1], "-b",
                numerical[0], ".r", numerical[1], ".r",
                ax=ax, markersize=15)
    ax.set(xlim=(1, 5), ylim=(0, 2),
           xticks=np.arange(1, 5.01, .5), yticks=np.arange(0, 2.01, .5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_system_pantograph_curve(exact, numerical):
    """Source ChebMatrix plot styles expanded over its two scalar blocks."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact[0], "-b", exact[1], "-b",
                numerical[0], ".--r", numerical[1], ".--r",
                ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(0, 1.01, .2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_vanishing_delay_curve(y):
    """Render the source vanishing-delay solution with reference axes."""
    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    points = jnp.linspace(y.domain.breakpoints[0], y.domain.breakpoints[-1], 1200)
    ax.plot(np.asarray(points), np.asarray(y(points)))
    ax.set(xlim=(0, y.domain.breakpoints[-1]), ylim=(0, 35),
           xticks=np.arange(0, 4.51, .5), yticks=np.arange(0, 36, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_continuous_delay_curve(exact, numerical):
    """Render the source continuous and pantograph delay plot with reference axes."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact, "-", numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 20), ylim=(0, 3),
           xticks=np.arange(0, 21, 5), yticks=np.arange(0, 3.01, .5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_interval_boundary_curve(numerical):
    """Render the source interior-boundary pantograph solution."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(numerical, "-", ax=ax)
    ax.set(xlim=(0, 1 / 3), ylim=(.258, .270),
           xticks=np.arange(0, .301, .05), yticks=np.arange(.258, .2701, .002))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_two_interval_delay_curve(exact, numerical):
    """Render the source two-interval constant-delay plot with reference axes."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact, "-", numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(1, 2.2),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(1, 2.21, .2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_second_order_delay_curve(exact, numerical):
    """Render the source second-order pantograph plot with reference axes."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact, "-", numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(0, 1.01, .2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_cubic_delay_curve(exact, numerical):
    """Render the source cubic state-delay plot with reference axes."""
    from chebfunjax.plotting import matlab_plot

    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    matlab_plot(exact, "-", numerical, ".--", ax=ax, markersize=15)
    ax.set(xlim=(0, 1), ylim=(-.2, 1),
           xticks=np.arange(0, 1.01, .2), yticks=np.arange(-.2, 1.01, .2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig


def _source_legacy_axes(image_id):
    """Scoped cached published axes, from native automatic plot limits.

    Chebfun examples f4b9ea46, DelayDifferentialEquations figures 1--6,
    12 and 14. These historical axes are specific to this page.
    """
    limits = {
        1: ((1, 3), np.arange(1, 3.01, .5)),
        2: ((-.5, 1), np.arange(-.5, 1.01, .5)),
        3: ((-.5, 1), np.arange(-.5, 1.01, .5)),
        4: ((-.4, 1), np.arange(-.4, 1.01, .2)),
        5: ((1, 3), np.arange(1, 3.01, .5)),
        6: ((1, 2.2), np.arange(1, 2.201, .2)),
        12: ((.86, 1), np.arange(.86, 1.001, .02)),
        14: ((.4, 1), np.arange(.4, 1.001, .1)),
    }
    ylim, yticks = limits[image_id]
    fig = plt.figure(figsize=(6, 2.7))
    ax = fig.add_axes([.13, .11, .775, .815])
    ax.set(xlim=(0, 1), ylim=ylim,
           xticks=np.arange(0, 1.01, .2), yticks=yticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    return fig, ax


def _plot_disc(t, x, sol, tt=None):
    """Original numeric line vertices and MarkerSize15 source point geometry."""
    from chebfunjax.plotting import _source_plot_line_and_points

    image_id = SOURCE_FIGURES[OUTPUT_SLOT[0]]
    fig, ax = _source_legacy_axes(image_id)
    if tt is None:
        # Source figure1 connects the original16 Chebyshev samples.
        tt = t if image_id == 1 else np.linspace(float(t[0]), float(t[-1]), 1001)
    s = sol(tt) if callable(sol) else sol
    ax.plot(np.asarray(tt), np.asarray(s), "-")
    _source_plot_line_and_points(
        ax, np.asarray(t), np.asarray(x), np.asarray(t), np.asarray(x),
        ".", {"markersize": 15})
    _save(fig, source_axes=True)


def _show(x, label="x"):
    """Display the actual public representation using the source label."""
    print(f"{label} =\n{x!r}")


def _plot_pair(sol, x):
    """Source FUN line grids and actual representation nodes, not markevery."""
    from chebfunjax.plotting import matlab_plot

    fig, ax = _source_legacy_axes(SOURCE_FIGURES[OUTPUT_SLOT[0]])
    matlab_plot(sol, "-", x, ".--", ax=ax, markersize=15)
    _save(fig, source_axes=True)


def _capture_figure_data(fig, image_id, source_locals):
    """Opt-in I/O snapshot after rendering; no evaluation or reconstruction.

    Artist coordinates support future style-only replay. Coefficients and
    explicit endpoint overrides are copied from current representations.
    """
    import hashlib
    import json
    from pathlib import Path

    root = Path(os.environ["CHEBFUNJAX_DDE_FIGURE_CAPTURE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    axes_metadata = []
    for ai, ax in enumerate(fig.axes):
        lines = []
        for li, artist in enumerate(ax.lines):
            key = f"axes{ai}_line{li}"
            arrays[key + "_xy"] = np.asarray(artist.get_xydata()).copy()
            marker = artist._marker
            arrays[key + "_marker_path"] = np.asarray(
                marker.get_path().transformed(marker.get_transform()).vertices).copy()
            codes = marker.get_path().codes
            if codes is not None:
                arrays[key + "_marker_codes"] = np.asarray(codes).copy()
            lines.append({
                "key": key, "class": type(artist).__name__,
                "color": artist.get_color(), "linestyle": artist.get_linestyle(),
                "linewidth": artist.get_linewidth(), "markersize": artist.get_markersize(),
                "marker": str(artist.get_marker()),
                "markerfacecolor": artist.get_markerfacecolor(),
                "markeredgecolor": artist.get_markeredgecolor(),
                "markeredgewidth": artist.get_markeredgewidth(),
                "clip_on": artist.get_clip_on(), "zorder": artist.get_zorder(),
                "label": artist.get_label(),
            })
        axes_metadata.append({
            "bounds_fraction": list(ax.get_position().bounds),
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "xticks": list(ax.get_xticks()), "yticks": list(ax.get_yticks()),
            "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "title": ax.get_title(), "lines": lines,
        })
    representations = []

    def copy_source(name, value):
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                copy_source(f"{name}_{i}", item)
        elif hasattr(value, "funs") and hasattr(value, "domain"):
            pieces = []
            for i, piece in enumerate(value.funs):
                key = f"representation_{name}_{i}"
                arrays[key + "_coeffs"] = np.asarray(piece.tech.coeffs).copy()
                pieces.append({"key": key, "interval": list(piece.interval),
                               "tech": type(piece.tech).__name__})
            override = getattr(value, "_point_values", None)
            if override is not None:
                arrays[f"representation_{name}_point_override"] = np.asarray(override).copy()
            representations.append({"name": name, "domain": list(value.domain.breakpoints),
                                    "explicit_point_override": override is not None,
                                    "is_transposed": value.is_transposed, "pieces": pieces})
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            arrays["source_local_" + name] = np.asarray(value).copy()

    for name, value in source_locals.items():
        copy_source(name, value)
    stem = f"DelayDifferentialEquations_{image_id:02d}"
    np.savez(root / (stem + ".npz"), **arrays)
    png = Path(_IMG) / (stem + ".png")
    metadata = {"figure": image_id, "source_slot": OUTPUT_SLOT[0],
                "figure_inches": list(fig.get_size_inches()), "dpi": fig.dpi,
                "axes": axes_metadata, "representations": representations,
                "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                "capture_scope": "current artists and stored source metadata; no callback evaluation"}
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    (root / (stem + ".json")).write_text(
        json.dumps(metadata, indent=2, default=json_scalar))


def _output_slot(index):
    """Identify the cached MATLAB output cell during a captured run."""
    OUTPUT_SLOT[0] = index
    if os.environ.get("CHEBFUNJAX_OUTPUT_SLOTS") == "1":
        print(f"@@CHEBFUNJAX_OUTPUT_SLOT:{index}@@", flush=True)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # ----- 1. An ODE by explicit spectral discretization --------------
    _output_slot(1)
    a, x0 = 1.0, 1.0
    dom = (0.0, 1.0)
    n = 16
    t = chebpts_dom(n, dom)
    D = np.asarray(diffmat(n, 1, dom), dtype=float)
    Iden = np.eye(n)
    A = D - a * Iden
    rhs = 0 * t
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = x0
    x = np.linalg.solve(A, rhs)
    sol = np.exp(a * t) * x0
    err = float(np.linalg.norm(x - sol))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t, x, lambda s: np.exp(a * s) * x0)

    # ----- 2. The pantograph equation, discretized --------------------
    _output_slot(2)
    a, b, q = 1.0, -8.0, 0.5
    tau = q * t
    E = np.asarray(barymat(jnp.asarray(tau), jnp.asarray(t)), dtype=float)
    A = D - a * Iden - b * E
    rhs = 0 * t
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = x0
    x = np.linalg.solve(A, rhs)
    sol_fn = lambda s: -7 / 2 * s**3 + 21 / 2 * s**2 - 7 * s + 1
    err = float(np.linalg.norm(x - sol_fn(t)))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t, x, sol_fn)

    # ----- 3. The pantograph equation, chebop --------------------------
    _output_slot(3)
    N = Chebop(lambda t, x: x.diff() - a * x - b * x(q * t), domain=dom)
    N.lbc = x0
    xc = N.solve(0.0)
    _show(xc)
    solc = chebfun(sol_fn, domain=dom)
    err = float((xc - solc).norm())
    print(f"err =\n     {err:.15e}")
    _plot_pair(solc, xc)

    # ----- 4. Two pantograph delays ------------------------------------
    _output_slot(4)
    q1, q2 = 1 / 2, 1 / 3
    N = Chebop(lambda t, x: x.diff() - a * x - b * x(q1 * t) - x(q2 * t),
               domain=dom)
    N.lbc = x0
    xc = N.solve(0.0)
    _show(xc)
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")
    from chebfunjax.plotting import matlab_plot
    fig, ax = _source_legacy_axes(4)
    matlab_plot(xc, ax=ax)
    _save(fig, source_axes=True)

    # ----- 5. A two-interval discretization (plain ODE) ---------------
    _output_slot(5)
    n = 10
    p = 0.5
    domL, domR = (0.0, p), (p, 2 * p)
    tL = chebpts_dom(n, domL)
    tR = chebpts_dom(n, domR)
    t2 = np.concatenate([tL, tR])
    DL = np.asarray(diffmat(n, 1, domL), dtype=float)
    DR = np.asarray(diffmat(n, 1, domR), dtype=float)
    Z = np.zeros((n, n))
    AL = np.hstack([DL - a * np.eye(n), Z])
    AR = np.hstack([Z, DR - a * np.eye(n)])
    B = np.zeros(2 * n)
    B[0] = 1
    C = np.zeros(2 * n)
    C[n - 1] = 1
    C[n] = -1
    A = np.vstack([B, AL[1:n, :], C, AR[1:n, :]])
    rhs = np.concatenate([[x0], np.zeros(n - 1), [0.0], np.zeros(n - 1)])
    x = np.linalg.solve(A, rhs)
    err = float(np.linalg.norm(x - np.exp(a * t2) * x0))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t2, x, lambda s: np.exp(a * s) * x0)

    # ----- 6. A constant delay, discretized ----------------------------
    _output_slot(6)
    tauL = np.maximum(tL - p, 0)
    tauR = tR - p
    EL = np.asarray(barymat(jnp.asarray(tauL), jnp.asarray(tL)), dtype=float)
    ER = np.asarray(barymat(jnp.asarray(tauR), jnp.asarray(tL)), dtype=float)
    AL = np.hstack([DL - a * EL, Z])
    AR = np.hstack([-ER, DR])
    A = np.vstack([B, AL[1:n, :], C, AR[1:n, :]])
    x = np.linalg.solve(A, rhs)
    solv = np.concatenate([a * tL + 1,
                           (a * p + 1) + a * (tR - p) * (.5 * a * (tR - p) + 1)])
    err = float(np.linalg.norm(x - solv))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t2, x, solv, tt=t2)

    # ----- 7. The same delay problem through chebop --------------------
    _output_slot(7)
    N = Chebop(lambda t, x: x.diff() - x((t - p).maximum(0.0)),
               domain=(0.0, 0.5, 1.0))
    N.lbc = 1.0
    xc = N.solve(0.0)
    print(f"x =\n{xc!r}")
    sol7 = chebfun([lambda t: t + 1,
                   lambda t: (a * p + 1) + a * (t - p) * (.5 * a * (t - p) + 1)],
                  domain=(0.0, 0.5, 1.0))
    err = float((xc - sol7).norm())
    print(f"err =\n     {err:.15e}")
    _save(_source_two_interval_delay_curve(sol7, xc), source_axes=True)

    # ----- 8. Longer domain: six pieces --------------------------------
    _output_slot(8)
    N = Chebop(lambda t, x: x.diff() - x((t - p).maximum(0.0)),
               domain=tuple(np.arange(0, 3.01, 0.5)))
    N.lbc = 1.0
    xc = N.solve(0.0)
    print("x =")
    print(repr(xc))
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")
    tt = np.linspace(0, 3, 1200)
    fig = _source_delay_curve(tt, np.asarray(xc(tt)), 8)
    _save(fig, source_axes=True)
    fig = _source_delay_spy(np.asarray(N.matrix(10)))
    _save(fig, source_axes=True, spy=True)

    # ----- 9. Two constant delays --------------------------------------
    _output_slot(9)
    p1, p2 = 1 / 2, 1 / 3
    from chebfunjax.utils.colon import matlab_colon

    dom9 = tuple(sorted(set(map(float, matlab_colon(0., p1, 2.)))
                            | set(map(float, matlab_colon(0., p2, 2.)))))
    N = Chebop(lambda t, x: x.diff() - x((t - p1).maximum(0.0))
               - x((t - p2).maximum(0.0)) - x, domain=dom9)
    N.lbc = 1.0
    xc = N.solve(0.0)
    print("x =")
    print(repr(xc))
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")
    tt = np.linspace(0, 2, 1200)
    fig = _source_delay_curve(tt, np.asarray(xc(tt)), 9)
    _save(fig, source_axes=True)
    fig = _source_delay_spy(np.asarray(N.matrix(10)))
    _save(fig, source_axes=True, spy=True)

    # ----- 10. A Volterra integro-differential equation ----------------
    _output_slot(10)
    f = lambda s, tv: np.exp(tv - s)
    n = 16
    t = chebpts_dom(n, dom)
    D = np.asarray(diffmat(n, 1, dom), dtype=float)
    av = 1 / 2
    # MATLAB's f(t, t') puts s down the rows: F_ij = f(t_i, t_j)
    # = exp(t_j - t_i)
    V = f(t[:, None], t[None, :]) * np.asarray(
        cumsummat(n, dom), dtype=float)
    A = D - V + av * np.eye(n)
    rhs = 0 * t
    A[0, :] = 0
    A[0, 0] = 1
    rhs[0] = x0
    x = np.linalg.solve(A, rhs)
    bb = np.sqrt(av**2 - 2 * av + 5) / 2
    sol10 = lambda s: np.exp(-(av + 1) / 2 * s) * (
        np.cosh(bb * s) + .5 * (1 - av) / bb * np.sinh(bb * s))
    err = float(np.linalg.norm(x - sol10(t)))
    print(f"err =\n     {err:.15e}")
    _plot_disc(t, x, sol10)

    # ----- 11. The same through chebop + volt --------------------------
    _output_slot(11)
    # our volt kernel is K(x, y) with (Kf)(x) = int K(x,y) f(y) dy;
    # the equation's kernel in that convention is exp(y - x)
    N = Chebop(lambda t, x: x.diff() + av * x
               - volt(lambda X, Y: jnp.exp(Y - X), x), domain=dom)
    N.lbc = x0
    xc = N.solve(0.0)
    print(f"x =\n{xc!r}")
    solc = chebfun(sol10, domain=dom)
    err = float((xc - solc).norm())
    print(f"err =\n     {err:.15e}")
    _save(_source_volterra_curve(solc, xc), source_axes=True)

    # ----- 12. A state-dependent delay, Newton by hand -----------------
    _output_slot(12)
    n = 16
    t = chebpts_dom(n, dom)
    D = np.asarray(diffmat(n, 1, dom), dtype=float)
    x = t.copy()
    for _j in range(10):
        E = np.asarray(barymat(jnp.asarray(x), jnp.asarray(t)), dtype=float)
        F = D @ x + E @ x
        J = D + E + np.diag(E @ D @ x)
        J[0, :] = 0
        J[0, 0] = 1
        F[0] = x[0] - x0
        dx = np.linalg.solve(J, F)
        x = x - dx
        ndu = float(np.linalg.norm(dx))
        print(f"ndu =\n   {ndu:.15g}")
        if ndu < 1e-10:
            break
    fig, ax = _source_legacy_axes(14)
    ax.plot(t, x)
    _save(fig, source_axes=True)

    from chebfunjax.chebfun1d.chebfun import Chebfun as _Chebfun
    from chebfunjax.domain import Domain as _Domain
    xc12 = _Chebfun.from_values(jnp.asarray(x), _Domain(dom))
    _output_slot(13)
    res = float((xc12.diff() + xc12(xc12)).norm())
    print(f"res =\n     {res:.15e}")

    # ----- 13. x' + x(x) = 0 through chebop ----------------------------
    _output_slot(14)
    N = Chebop(lambda t, x: x.diff() + x(x), domain=dom)
    N.lbc = 1.0
    xc = N.solve(0.0)
    print(f"x =\n{xc!r}")
    res = float(N(xc).norm())
    print(f"res =\n     {res:.15e}")

    _save(_source_state_delay_curve(t, x, xc), source_axes=True)

    # ----- 14. A pantograph problem with known solution ----------------
    _output_slot(15)
    q = 0.5
    tcf = chebfun(lambda t: t, domain=dom)
    g = -2 * tcf / (1 + tcf**2)**2 - 1 / (1 + (q * tcf)**2)**2
    N = Chebop(lambda t, y: y.diff() - y(q * t)**2, domain=dom)
    N.lbc = 1.0
    y = N.solve(g)
    print(f"y =\n{y!r}")
    exact = 1 / (1 + tcf**2)
    err = float((y - exact).norm())
    print(f"err =\n     {err:.15e}")
    _save(_source_pantograph_curve(exact, y), source_axes=True)

    # ----- 15. A delayed system ----------------------------------------
    _output_slot(16)
    q = 0.7
    g1 = (tcf.cos() - (q * tcf).cos() * tcf.cos()
          - (q * tcf).sin() * tcf.sin())
    g2 = ((q * tcf).sin() * tcf.cos() - (q * tcf).cos() * tcf.sin()
          - tcf.sin())
    N = Chebop(lambda t, u, v: [
        u.diff() - (t.sin() * u(q * t) + t.cos() * v(q * t)),
        v.diff() - (-t.cos() * u(q * t) + t.sin() * v(q * t))],
        domain=dom)
    N.lbc = [jnp.sin(0.0), jnp.cos(0.0)]
    u, v = N.solve([g1, g2])
    from chebfunjax.operators.chebmatrix import ChebMatrix
    exact = [tcf.sin(), tcf.cos()]
    _source_system_display("y", [u, v])
    _source_system_display("sol", exact)
    err = ChebMatrix([[u - exact[0]], [v - exact[1]]]).norm()
    print(f"err =\n     {err:.15e}")
    _save(_source_system_pantograph_curve(exact, [u, v]), source_axes=True)

    # ----- 16. A state-dependent y(y) with known solution --------------
    _output_slot(17)
    xcf = chebfun(lambda x: x, domain=dom)
    mu = 0.0
    g = (3 + mu) * xcf**(2 + mu) - xcf**((3 + mu)**2)
    N = Chebop(lambda t, y: y.diff() - y(y), domain=dom)
    N.lbc = 0.0
    y = N.solve(g)
    print(f"y =\n{y!r}")
    exact = xcf**(3 + mu)
    err = float((y - exact).norm())
    print(f"err =\n     {err:.15e}")
    _save(_source_cubic_delay_curve(exact, y), source_axes=True)

    # ----- 17. State-dependent 1/t cascade ----------------------------
    _output_slot(18)
    from chebfunjax.utils._matlab_fzero import _fzero_default

    dom = [0., float((1 + jnp.sqrt(jnp.asarray(5.))) / 2)]
    for _ in range(10):
        previous = dom[-1]
        root, _info = _fzero_default(lambda x: x - 1 / x - previous, previous)
        dom.append(float(root))
    tcf = chebfun(lambda x: x, domain=dom)
    N = Chebop(lambda t, y: y.diff() - y((t - 1 / t).maximum(0.)), domain=dom)
    N.lbc = 1.
    N.init = 1 + tcf
    y = N.solve(0.)
    print(f"y =\n{y!r}")
    res = float(N(y).norm())
    print(f"res =\n     {res:.15e}")
    _save(_source_vanishing_delay_curve(y), source_axes=True)

    # ----- 18. Delay + integral terms together ------------------------
    _output_slot(19)
    dom20 = (0.0, 20.0)
    q = 0.5

    def op18(x, u):
        return (u.diff() - 1 / 100 * (q * x - x - 10) * u(q * x)
                - 1 / 100 * (x + 20) * jnp.exp(-1.)
                - 1 / 100 * u.cumsum()
                - 1 / 1000 * volt(lambda t, s: t / q - s, u)(q * x))

    N = Chebop(op18, domain=dom20)
    N.lbc = jnp.exp(-1.)
    u18 = N.solve(0.)
    print(f"u =\n{u18!r}")
    sol18 = chebfun(lambda x: jnp.exp(x / 10 - 1), domain=dom20)
    err = float((sol18 - u18).norm())
    print(f"err =\n     {err:.15e}")
    _save(_source_continuous_delay_curve(sol18, u18), source_axes=True)

    # ----- 19. A second-order equation with delays --------------------
    _output_slot(20)
    dom = (0., 1.)
    tcf = chebfun(lambda x: x, domain=dom)
    N = Chebop(lambda t, y: y.diff(2) + y(t) - 5 * y(t / 2)**2,
               domain=dom)
    N.lbc = lambda u: [u - 1, u.diff() + 2]
    N.init = 1 + tcf
    y19 = N.solve(0.0)
    print(f"y =\n{y19!r}")
    sol19 = (-2 * tcf).exp()
    err = float((y19 - sol19).norm())
    print(f"err =\n     {err:.15e}")
    _save(_source_second_order_delay_curve(sol19, y19), source_axes=True)

    # ----- 20. An interval condition u(0) = u(1/3) --------------------
    _output_slot(21)
    dom3 = (0.0, 1 / 3)
    t3 = chebfun(lambda t: t, domain=dom3)
    g = t3.exp() / 2
    N = Chebop(lambda t, u: u.diff() + u**2 + 2 * u(t / 2), domain=dom3)
    N.bc = lambda t, u: u(0.0) - u(1 / 3)
    N.init = 1
    u20 = N.solve(g)
    print(f"u =\n{u20!r}")
    differential = float((N(u20) - g).norm())
    boundary = float(jnp.abs(N.bc(t3, u20)))
    res = float(jnp.sqrt(differential**2 + boundary**2))
    print(f"res =\n     {res:.15e}")
    _save(_source_interval_boundary_curve(u20), source_axes=True)

    # ----- 21. Three-variable constant-delay system -------------------
    _output_slot(22)
    from chebfunjax.utils.native_ddesd import ddesd
    dom22 = tuple(float(v) for v in matlab_colon(0., .2, 2.))
    lag22 = (1., .2)
    N = Chebop(lambda t, x, y, z: [
        x.diff() - x((t - lag22[0]).maximum(0.)),
        y.diff() - x((t - lag22[0]).maximum(0.)) - y((t - lag22[1]).maximum(0.)),
        z.diff() - y], domain=dom22)
    N.lbc = [1., 1., 1.]
    components22 = list(N.solve(0.))
    x22 = components22[0]
    _show(x22, "x")
    sol22 = ddesd(_source_ddex1de, [1., .2], _source_ddex1hist,
                  [0., dom22[-1]], {"AbsTol": 1e-12, "RelTol": 1e-12})
    err = float(jnp.linalg.norm(x22(sol22.x) - sol22.y[0]))
    print(f"err =\n     {err:.15e}")
    _save(_source_discrete_delay_system_curve(
        components22, sol22.x, sol22.y), source_axes=True)

    # ----- 22. A system with a state-dependent delay ------------------
    _output_slot(23)
    dom5 = (1.0, 5.0)
    t5 = chebfun(lambda t: t, domain=dom5)
    N = Chebop(lambda t, u, v: [
        u.diff() - v,
        v.diff() + v((1 - v).exp()) * v**2 * (1 - v).exp()], domain=dom5)
    N.lbc = [np.log(1.0), 1.0]
    u22, v22 = N.solve(0.0)
    _show(u22, "ans")
    from chebfunjax.operators.chebmatrix import ChebMatrix
    exact = [t5.log(), 1 / t5]
    err = ChebMatrix([[u22 - exact[0]], [v22 - exact[1]]], domain=dom5).norm()
    print(f"err =\n     {err:.15e}")
    _save(_source_state_dependent_system_curve(exact, [u22, v22]), source_axes=True)

    _output_slot(0)


if __name__ == "__main__":
    run()
