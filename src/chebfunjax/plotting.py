"""Plotting utilities for chebfunjax.

Provides MATLAB-Chebfun-style plotting functions for Chebfun objects.
All functions return (fig, ax) so callers can overlay additional plots
or customise the figure before saving.

Style constants match the Chebfun blue used by the MATLAB documentation.

New in this version
-------------------
- :func:`plot` — now accepts multiple Chebfuns as positional arguments for
  overlaying, and a ``title`` keyword to set the axes title.
- :func:`waterfall` — waterfall/cascade plot for a sequence of Chebfuns
  (e.g. time snapshots).
- :func:`roots_plot` — plot a Chebfun with its roots marked as red circles.
- :func:`spy` — sparsity pattern for operator matrices (wraps matplotlib spy).
- :func:`plotregion` — Bernstein ellipse showing the region of analyticity.
- :func:`arrowplot` — parametric curve with direction arrows (complex chebfun).
- :func:`chebpolyplot` — Chebyshev coefficient magnitudes with log scale and
  envelope line (enhanced version of plotcoeffs).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib
import matplotlib as mpl

# Do NOT call matplotlib.use("Agg") unconditionally — that breaks Jupyter
# inline plotting.  Only switch if we are already headless or truly have no
# display available.
if matplotlib.get_backend().lower() == "agg" and not os.environ.get("DISPLAY"):
    pass  # already headless — keep whatever backend is active
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: matplotlib rendering interop (host-side, never in JIT paths)
from matplotlib.colors import LightSource, Normalize

from chebfunjax.utils.quadrature import chebpts, trigpts

# ---------------------------------------------------------------------------
# Chebfun RC style (Chebfun-quality plots)
# ---------------------------------------------------------------------------

CHEBFUN_RC = {
    # Match MATLAB Chebfun default plot style exactly
    'figure.figsize': (6.1, 2.58),       # MATLAB default aspect ratio ~2.4:1
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.linewidth': 0.5,               # thin box
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.grid': False,                   # MATLAB Chebfun: NO grid
    'axes.spines.top': True,              # MATLAB: full box
    'axes.spines.right': True,            # MATLAB: full box
    'axes.xmargin': 0.0,                  # tight x-limits like MATLAB
    'axes.ymargin': 0.05,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',             # MATLAB default: inward ticks
    'ytick.direction': 'in',
    'lines.linewidth': 1.2,              # MATLAB default ~1.0-1.5
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.dpi': 150,
    'axes.prop_cycle': mpl.cycler(color=[
        '#0072BD',  # MATLAB blue (default)
        '#D95319',  # MATLAB orange
        '#EDB120',  # MATLAB yellow
        '#7E2F8E',  # MATLAB purple
        '#77AC30',  # MATLAB green
        '#4DBEEE',  # MATLAB cyan
        '#A2142F',  # MATLAB dark red
    ]),
}


def chebfun_style():
    """Apply Chebfun plot style globally."""
    mpl.rcParams.update(CHEBFUN_RC)


def save_chebfun_figure(fig, path, size=(600, 270)):
    """Save *fig* at an exact pixel size matching chebfun.org renders.

    The figures published on chebfun.org use fixed canvas sizes
    (600x270 px for examples, 610x258 px for the Guide chapters).
    Matplotlib's ``bbox_inches='tight'`` rescales the canvas to the
    content, which breaks pixel-level comparison against those
    references — so this helper pins the canvas instead.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Output PNG path.
    size : tuple of int
        Target (width, height) in pixels. Defaults to the chebfun.org
        example figure size; pass ``(610, 258)`` for Guide figures.
    """
    w, h = size
    dpi = 100.0
    fig.set_size_inches(w / dpi, h / dpi)
    # rc 'savefig.bbox: tight' would rescale the canvas even when
    # bbox_inches is not passed — force it off for the exact-size export.
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(path, dpi=dpi, facecolor="white")


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CHEBFUN_BLUE = "#0072BD"   # MATLAB default blue
CHEBFUN_RED  = "#D95319"   # MATLAB default orange/red
CHEBFUN_GREEN = "#77AC30"  # MATLAB default green
CHEBFUN_ORANGE = "#EDB120" # MATLAB default yellow/orange

_DEFAULT_LINE_KW: dict[str, Any] = dict(color=CHEBFUN_BLUE, linewidth=1.2)
_DEFAULT_GRID_KW: dict[str, Any] = dict(alpha=0.3, linestyle="--", linewidth=0.6)


def _matlab_ticks(ax: plt.Axes) -> None:
    """MATLAB-like tick density and label format on linear axes.

    MATLAB picks sparser ticks than matplotlib's default (e.g. steps of
    0.5 on [-1, 1] where matplotlib chooses 0.25) and prints labels
    without trailing zeros (``0.5``, not ``0.50``). Log-scale axes are
    left untouched.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for axis, scale in ((ax.xaxis, ax.get_xscale()),
                        (ax.yaxis, ax.get_yscale())):
        if scale == "linear":
            # nbins=7 measured empirically against the chebfun.org MATLAB
            # renders across guide chapter 1: it reproduces MATLAB's choices
            # (0.5 on [-1,1], 0.2 on [0,1.2], integers on [0,3]) where
            # nbins=5 comes out too sparse on several of those ranges.
            axis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _apply_style(ax: plt.Axes, title: str = "", xlabel: str = "",
                 ylabel: str = "", grid: bool = False) -> None:
    """Apply MATLAB-Chebfun style to an Axes: no grid, full box, no labels."""
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if grid:
        ax.grid(True, **_DEFAULT_GRID_KW)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    _matlab_ticks(ax)


def _domain_points(f, n: int = 600) -> np.ndarray:
    """Return *n* equispaced points spanning the domain of a Chebfun."""
    # Works for both Chebfun1D (f.domain.breakpoints) and plain Chebfun2.
    try:
        bp = f.domain.breakpoints
        a, b = float(bp[0]), float(bp[-1])
        # Unbounded domains: plot on a finite window like MATLAB
        # (10 units past the finite endpoint, or [-10, 10]).
        if np.isinf(a) and np.isinf(b):
            a, b = -10.0, 10.0
        elif np.isinf(b):
            b = a + 10.0
        elif np.isinf(a):
            a = b - 10.0
        return np.linspace(a, b, n)
    except AttributeError:
        return np.linspace(-1.0, 1.0, n)


def _eval_2d_vectorized(f2, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """Evaluate a 2-D function on a grid, using vectorized evaluation.

    Tries passing the full 2-D arrays first (fast path for Chebfun2 which
    accepts arbitrary-shape inputs).  On failure, falls back to ravelled
    1-D arrays.  If that also fails, falls back to a Python double loop.
    """
    import jax.numpy as jnp

    # Fast path: pass 2-D arrays directly.
    try:
        ZZ = np.array(f2(jnp.array(XX), jnp.array(YY)))
        if ZZ.shape == XX.shape:
            return ZZ
    except Exception:
        pass

    # Ravelled 1-D path.
    try:
        xflat = jnp.array(XX.ravel())
        yflat = jnp.array(YY.ravel())
        ZZ = np.array(f2(xflat, yflat)).reshape(XX.shape)
        return ZZ
    except Exception:
        pass

    # Scalar fallback — slow but always correct.
    ZZ = np.empty(XX.shape, dtype=float)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = float(f2(jnp.array(XX[i, j]), jnp.array(YY[i, j])))
    return ZZ


# ---------------------------------------------------------------------------
# Custom parula colormap (MATLAB default)
# ---------------------------------------------------------------------------

def _make_parula_cmap():
    """Build a close approximation of MATLAB's parula colormap.

    Parula goes from dark blue -> teal -> yellow.  This 9-anchor
    linear-segment approximation is visually indistinguishable from the
    real thing at typical monitor resolutions.
    """
    from matplotlib.colors import LinearSegmentedColormap

    _parula_data = [
        (0.2422, 0.1504, 0.6603),
        (0.2810, 0.3228, 0.9579),
        (0.1786, 0.5289, 0.9682),
        (0.0689, 0.6948, 0.8394),
        (0.1280, 0.7890, 0.5920),
        (0.4676, 0.7804, 0.3723),
        (0.7914, 0.7314, 0.1725),
        (0.9763, 0.8312, 0.0538),
        (0.9769, 0.9839, 0.0805),
    ]
    return LinearSegmentedColormap.from_list("parula", _parula_data, N=256)


PARULA = _make_parula_cmap()


def _coerce_cmap(cmap):
    """Return a matplotlib colormap object."""
    if cmap is None:
        return PARULA
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    return cmap


def _normalize_values(values: np.ndarray) -> Normalize:
    """Normalization with a stable fallback for near-constant data."""
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    elif vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def _matlab_facecolors(
    values: np.ndarray,
    cmap,
    shade_data: np.ndarray | None = None,
    *,
    apply_lighting: bool = False,
) -> np.ndarray:
    """Map scalar values to RGBA facecolors, optionally with headlight shading."""
    cmap_obj = _coerce_cmap(cmap)
    norm = _normalize_values(values)
    rgba = cmap_obj(norm(values))
    if apply_lighting and shade_data is not None:
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade_rgb(rgba[:, :, :3], shade_data)
        rgba = rgba.copy()
        rgba[:, :, :3] = rgb
    return rgba


def _draw_disk_boundary(ax: plt.Axes, *, dashed: bool = False, linewidth: float = 0.5) -> None:
    """Draw the unit-circle outline used by MATLAB disk plots."""
    t = np.linspace(-np.pi, np.pi, 201)
    ax.plot(np.cos(t), np.sin(t), "k--" if dashed else "k-", linewidth=linewidth)


def _draw_sphere_background(
    ax,
    *,
    color=(255 / 255, 255 / 255, 204 / 255),
    scale: float = 0.99,
    alpha: float = 1.0,
) -> None:
    """Draw the slightly shrunken background sphere used by MATLAB quiver/contour plots."""
    u = np.linspace(0.0, 2.0 * np.pi, 102)
    v = np.linspace(0.0, np.pi, 52)
    xs = scale * np.outer(np.cos(u), np.sin(v))
    ys = scale * np.outer(np.sin(u), np.sin(v))
    zs = scale * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        xs,
        ys,
        zs,
        color=color,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=alpha,
    )


def _setup_3d_axes(ax, fig, elev=30, azim=-127.5, figsize=(6.1, 2.58),
                   fill_canvas=True):
    """Create or configure 3D axes with MATLAB-Chebfun styling.

    Parameters
    ----------
    ax : Axes3D or None
        Existing axes, or None to create a new figure.
    fig : Figure or None
    elev, azim : float
        Camera view angles.
    figsize : tuple
        Figure size if creating a new figure.
    fill_canvas : bool
        If True (default), place the 3D axes with a slight canvas overflow so
        surf/height-field renders fill the frame like MATLAB's published
        figures. Sphere plots (viewed near the equator with a unit bounding
        box) must NOT fill the canvas — the overflow combined with
        ``bbox_inches='tight'`` blows up the crop and mis-frames the sphere
        (see the a1af714 camera-fix regression). Those callers pass
        ``fill_canvas=False`` to keep the plain ``add_subplot`` framing that
        the reference sphere renders were produced with.

    Returns
    -------
    fig, ax
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if fill_canvas:
            # Fill the canvas like MATLAB's published surf renders —
            # add_subplot leaves large margins around 3D axes.
            ax = fig.add_axes([0.02, -0.07, 0.96, 1.14], projection="3d")
        else:
            ax = fig.add_subplot(111, projection="3d")
    else:
        if fig is None:
            fig = ax.get_figure()

    ax.view_init(elev=elev, azim=azim)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Light gray grid lines (MATLAB style)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.15))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.15))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.15))
    ax.xaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.15)
    ax.yaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.15)
    ax.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.15)

    # Thin box edges
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.4)

    # No axis labels (MATLAB default for 3D Chebfun plots)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    return fig, ax


def _set_unit_ticks(ax, domain=None):
    """Set tick marks to -1, -0.5, 0, 0.5, 1 for unit domain axes."""
    unit_ticks = [-1, -0.5, 0, 0.5, 1]
    if domain is None:
        ax.set_xticks(unit_ticks)
        ax.set_yticks(unit_ticks)
    else:
        x0, x1, y0, y1 = domain
        if abs(x0 - (-1)) < 0.01 and abs(x1 - 1) < 0.01:
            ax.set_xticks(unit_ticks)
        if abs(y0 - (-1)) < 0.01 and abs(y1 - 1) < 0.01:
            ax.set_yticks(unit_ticks)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# 1-D plot
# ---------------------------------------------------------------------------

def plot_1d(
    *args,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    label: str = "",
    color: str = CHEBFUN_BLUE,
    linestyle: str = "-",
    linewidth: float = 1.2,
    n_pts: int = 600,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more 1-D Chebfuns on their domain.

    Parameters
    ----------
    *args : Chebfun or (Chebfun, Chebfun, ...)
        One or more Chebfuns to plot (overlaid on the same axes).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when not provided.
    title, xlabel, ylabel : str
        Axis labels/title.
    label : str
        Line label (for legends).  Applied to the first Chebfun only when
        overlaying multiple functions.
    color, linestyle, linewidth : plot style.
        Applied to the first Chebfun.  Additional Chebfuns cycle through a
        colour sequence.
    n_pts : int
        Number of evaluation points.

    Returns
    -------
    fig, ax

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import chebfunjax as cj
    >>> f = cj.chebfun(jnp.sin)
    >>> fig, ax = cj.plot(f)
    >>> fig, ax = cj.plot(f, f.diff(), title="sin and cos")
    """
    import jax.numpy as jnp

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    # Cycle of colours for multiple overlaid functions
    _colors = [CHEBFUN_BLUE, CHEBFUN_RED, CHEBFUN_GREEN, CHEBFUN_ORANGE,
               "#8B008B", "#008080"]

    for idx, f in enumerate(args):
        c = color if idx == 0 else _colors[idx % len(_colors)]
        plot_kw: dict[str, Any] = dict(color=c, linewidth=linewidth,
                                       linestyle=linestyle)
        if label and idx == 0:
            plot_kw["label"] = label
        # Forward extra kwargs only to first series to avoid clashes
        if idx == 0:
            plot_kw.update(kw)

        # Complex-valued chebfun: MATLAB plots the image curve in the
        # complex plane (real vs imag) with equal axis scaling. Sample
        # per piece so many-piece paths (e.g. scribble text) keep their
        # sharp corners instead of being corner-cut by one global grid.
        probe = np.array(f(jnp.array(_domain_points(f, 3))))
        if np.iscomplexobj(probe):
            funs_c = getattr(f, "funs", None)
            if funs_c is not None and len(funs_c) > 1:
                m = max(8, n_pts // len(funs_c))
                chunks = []
                for piece in funs_c:
                    pa, pb = (float(v) for v in piece.interval)
                    ts = np.linspace(pa, pb, m)
                    chunks.append(np.array(piece(jnp.array(ts))))
                ys = np.concatenate(chunks)
            else:
                xs = _domain_points(f, n_pts)
                ys = np.array(f(jnp.array(xs)))
            ax.plot(np.real(ys), np.imag(ys), **plot_kw)
            ax.set_aspect("equal")
            continue

        funs = getattr(f, "funs", None)
        if funs is not None and len(funs) > 1:
            # Piecewise chebfun: draw each smooth piece on its own grid so
            # jumps are not smeared into near-vertical solid segments, and
            # connect jump discontinuities with dotted verticals (MATLAB
            # Chebfun's plot style).
            total = _domain_points(f, 2)
            total_len = float(total[-1] - total[0]) or 1.0
            piece_ends: list[tuple[float, float, float]] = []
            for j, piece in enumerate(funs):
                a, b = (float(v) for v in piece.interval)
                m = max(16, int(round(n_pts * (b - a) / total_len)))
                xs = np.linspace(a, b, m)
                ys = np.array(piece(jnp.array(xs)))
                pk = dict(plot_kw)
                if j > 0:
                    pk.pop("label", None)
                ax.plot(xs, ys, **pk)
                piece_ends.append((a, float(ys[0]), float(ys[-1])))
            # Dotted connectors at interior breakpoints with a visible jump
            vsc = max(abs(e) for _, s, e in piece_ends) or 1.0
            for j in range(1, len(piece_ends)):
                xb = piece_ends[j][0]
                left = piece_ends[j - 1][2]
                right = piece_ends[j][1]
                if abs(right - left) > 1e-10 * vsc:
                    ax.plot([xb, xb], [left, right], linestyle=":",
                            color=c, linewidth=linewidth)
        else:
            xs = _domain_points(f, n_pts)
            ys = np.array(f(jnp.array(xs)))
            ax.plot(xs, ys, **plot_kw)

    _apply_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# Backward-compatibility alias: direct imports of ``plot`` from this module
# still resolve to the 1-D function, matching the original API.
plot = plot_1d


# ---------------------------------------------------------------------------
# Coefficient magnitude (semilogy)
# ---------------------------------------------------------------------------

def plotcoeffs(
    f,
    ax: Optional[plt.Axes] = None,
    title: str = "Chebyshev coefficients",
    color: str = CHEBFUN_BLUE,
    envelope: bool = True,
    loglog: bool = False,
    fmt: Optional[str] = None,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Semilogy plot of |Chebyshev coefficients| of *f*.

    Dots are plotted for each coefficient magnitude; an optional running
    maximum envelope line (matching MATLAB Chebfun's ``plotcoeffs`` style)
    is overlaid in a lighter colour.

    Parameters
    ----------
    f : Chebfun
        1-D Chebfun.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str
        Plot title.
    color : str
        Colour of the dots.
    envelope : bool, optional
        If ``True`` (default) overlay a running-max envelope line.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    # Array-valued chebfun / quasimatrix (list of columns) and piecewise
    # chebfuns: plot each column / smooth piece (MATLAB overlays them).
    cols = _cheb_cols(f) if not hasattr(f, "funs") else [f]
    series = []
    for c in (cols or [f]):
        funs = getattr(c, "funs", None)
        if funs is not None and len(funs) > 1:
            series.extend(np.abs(np.asarray(p.tech.coeffs)) for p in funs)
        else:
            series.append(np.abs(np.array(c.coeffs)))
    plot_fn = ax.loglog if loglog else ax.semilogy
    for s in series[1:]:
        if fmt:
            plot_fn(np.arange(len(s)), s, fmt, **kw)
        else:
            plot_fn(np.arange(len(s)), s, ".", markersize=4, **kw)
    coeffs = series[0]
    ns = np.arange(len(coeffs))

    if fmt:
        plot_fn(ns, coeffs, fmt, color=color, **kw)
    else:
        plot_fn(ns, coeffs, ".", color=color, markersize=4, **kw)

    if envelope and len(coeffs) > 2:
        # Running maximum from right (decaying coefficients show machine eps level)
        running_max = np.maximum.accumulate(coeffs[::-1])[::-1]
        ax.semilogy(ns, running_max, "-", color=color, alpha=0.3, linewidth=1.0)

    _apply_style(ax, title=title, xlabel="degree $n$", ylabel="$|a_n|$")
    ax.set_ylim(bottom=max(coeffs.min() * 0.1, 1e-18))
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2-D surface plot
# ---------------------------------------------------------------------------

def surf(
    f2,
    g2=None,
    h2=None,
    ax=None,
    title: str = "",
    n_pts: int = 100,
    cmap=None,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Surface plot of a Chebfun2 (MATLAB Chebfun style).

    Renders a smooth surface with the parula colormap, no axis labels,
    light gray grid, and thin box edges -- matching MATLAB's surf(f) output.

    Parameters
    ----------
    f2 : Chebfun2
        The 2-D function to plot.
    ax : Axes3D, optional
        3-D axes.  A new figure is created when not provided.
    title : str
        Plot title.
    n_pts : int
        Grid resolution per axis.
    cmap : colormap, optional
        Defaults to parula.

    Returns
    -------
    fig, ax
    """
    if cmap is None:
        cmap = PARULA

    # Determine domain
    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = _eval_2d_vectorized(f2, XX, YY)

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5,
                             figsize=(6.1, 2.58))

    if g2 is not None and h2 is not None:
        # Parametric surface surf(x, y, f): coordinates from the three
        # chebfun2s (MATLAB @separableApprox/surf.m).
        Xp = _eval_2d_vectorized(f2, XX, YY)
        Yp = _eval_2d_vectorized(g2, XX, YY)
        Zp = _eval_2d_vectorized(h2, XX, YY)
        ax.plot_surface(Xp, Yp, Zp, cmap=cmap,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True,
                        shade=True, **kw)
        _set_unit_ticks(ax, domain=(x0, x1, y0, y1))
        if title:
            ax.set_title(title, fontsize=10, pad=0)
        fig.tight_layout(pad=0.5)
        return fig, ax
    if g2 is not None:
        # surf(f, g): height from f, colouring from g (MATLAB).
        CC = _eval_2d_vectorized(g2, XX, YY)
        cmap_obj2 = _coerce_cmap(cmap)
        norm = _normalize_values(CC)
        ax.plot_surface(XX, YY, ZZ,
                        facecolors=cmap_obj2(norm(CC)),
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True,
                        shade=False, **kw)
        _set_unit_ticks(ax, domain=(x0, x1, y0, y1))
        if title:
            ax.set_title(title, fontsize=10, pad=0)
        fig.tight_layout(pad=0.5)
        return fig, ax

    ax.plot_surface(XX, YY, ZZ, cmap=cmap,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True,
                    shade=True, **kw)

    _set_unit_ticks(ax, domain=(x0, x1, y0, y1))

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# 2-D contour plot
# ---------------------------------------------------------------------------

def contour(
    f2,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    n_pts: int = 150,
    levels: int = 12,
    cmap=None,
    filled: bool = False,
    line_color=None,
    colorbar: bool = False,
    figsize: tuple = (6.1, 2.58),
    pivots=None,
    xx=None,
    yy=None,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Contour plot of a Chebfun2 (MATLAB Chebfun style).

    ``pivots`` takes a MATLAB linespec and overlays the GE pivot
    locations (MATLAB ``contour(f, 'pivots', S)``); ``xx``/``yy`` give
    an explicit evaluation grid (MATLAB ``contour(xx, yy, f)``).

    Draws contour lines (optionally over filled bands) using the parula
    colormap and unit-domain ticks.

    Parameters
    ----------
    f2 : Chebfun2
    ax : Axes, optional
    title : str
    n_pts : int
    levels : int
    cmap : colormap, optional  (default: parula)
    filled : bool
        If True, draw filled bands (contourf) under the contour lines.
        Default False (lines only).
    line_color : color spec, optional
        Colour for the overlaid contour lines.  Default None uses the
        colormap (guide-chapter style).  Pass e.g. ``"k"`` for the black
        contour lines the filled chebfun2 reference renders use.
    colorbar : bool
        If True, attach a colorbar to the filled bands (needs ``filled``).
        Default False.
    figsize : tuple
        Figure size when creating a new figure.  Default (6.1, 2.58) is the
        wide guide-chapter box; filled square renders (e.g. chebfun2_basics)
        pass a squarer size.

    Returns
    -------
    fig, ax
    """
    cmap_obj = _coerce_cmap(cmap)

    # MATLAB convention: contour(f, [v v]) draws the single level v.
    if not np.isscalar(levels):
        _lv = np.atleast_1d(np.asarray(levels, dtype=float))
        if _lv.size == 2 and _lv[0] == _lv[1]:
            levels = [float(_lv[0])]

    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    if xx is not None and yy is not None:
        XX = np.asarray(xx, dtype=float)
        YY = np.asarray(yy, dtype=float)
    else:
        xs = np.linspace(float(x0), float(x1), n_pts)
        ys = np.linspace(float(y0), float(y1), n_pts)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = _eval_2d_vectorized(f2, XX, YY)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cf = None
    if filled:
        lv_f = levels
        if not np.isscalar(lv_f):
            lv_arr = np.atleast_1d(np.asarray(lv_f, dtype=float))
            if lv_arr.size == 1:
                # matplotlib needs >= 2 levels for filled contours; fill
                # the region above the single MATLAB level.
                top = float(np.nanmax(ZZ))
                lv_f = [float(lv_arr[0]),
                        max(top, float(lv_arr[0]) + 1e-12)]
        cf = ax.contourf(XX, YY, ZZ, levels=lv_f, cmap=cmap_obj, **kw)
    if line_color is None:
        ax.contour(XX, YY, ZZ, levels=levels, cmap=cmap_obj,
                   linewidths=0.8, **kw)
    else:
        ax.contour(XX, YY, ZZ, levels=levels, colors=line_color,
                   linewidths=0.5, **kw)

    if pivots is not None:
        locs = np.asarray([(float(p[0]), float(p[1]))
                           for p in getattr(f2, "pivot_locations", ())],
                          dtype=float)
        if locs.size:
            fmt_p = pivots if isinstance(pivots, str) else "r."
            ax.plot(locs[:, 0], locs[:, 1], fmt_p)

    has_colorbar = colorbar and cf is not None
    if has_colorbar:
        fig.colorbar(cf, ax=ax)

    # With a colorbar the reference render lets the plot fill the box; a
    # forced equal aspect just shrinks it and leaves whitespace. Without a
    # colorbar (guide-chapter style) keep the square aspect.
    if not has_colorbar:
        ax.set_aspect("equal")
    _set_unit_ticks(ax, domain=(x0, x1, y0, y1))
    _apply_style(ax, title=title, grid=False)
    fig.set_facecolor("white")
    # tight_layout fights an attached colorbar (it re-flows the axes and
    # blows up the bbox_inches='tight' crop); skip it in that case.
    if not has_colorbar:
        fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Phase plot (complex functions)
# ---------------------------------------------------------------------------

def phaseplot(
    f,
    region=None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    n_pts: int = 500,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Phase portrait of a complex-valued function.

    Wraps :func:`chebfunjax.utils.phaseplot.phaseplot` and displays the
    resulting RGB image on a matplotlib Axes, returning ``(fig, ax)``.

    Parameters
    ----------
    f : callable
        Complex-valued function of a complex variable.
    region : sequence of 4 floats, optional
        ``[x_min, x_max, y_min, y_max]`` for the plot window.
        Defaults to ``[-1, 1, -1, 1]``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when not provided.
    title : str
        Plot title.
    n_pts : int
        Grid resolution.

    Returns
    -------
    fig, ax
    """
    from chebfunjax.utils.phaseplot import phaseplot as _phaseplot_impl

    if region is None:
        region = [-1.0, 1.0, -1.0, 1.0]

    img = _phaseplot_impl(f, ax=region, n_pts=n_pts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    x_min, x_max, y_min, y_max = region
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max],
              origin="lower", aspect="equal", **kw)
    _apply_style(ax, title=title, xlabel="Re", ylabel="Im", grid=False)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Disk function plot (polar heatmap)
# ---------------------------------------------------------------------------

def plot_disk(
    fd,
    ax=None,
    title: str = "",
    n_theta: int = 200,
    n_r: int = 100,
    cmap=None,
    mode: str = "3d",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Plot a Diskfun on the unit disk (MATLAB Chebfun style).

    In '3d' mode (default, matching MATLAB), renders the function values
    as a 3-D surface height over the disk with parula colormap.
    In '2d' mode, renders a flat pseudocolor plot.

    Parameters
    ----------
    fd : Diskfun
    ax : Axes, optional
    title : str
    n_theta, n_r : int
    cmap : colormap, optional (default: parula)
    mode : str
        '3d' (default) for surface plot, '2d' for flat pcolormesh.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=True)
    r = np.linspace(0.0, 1.0, n_r)
    TT, RR = np.meshgrid(theta, r, indexing="ij")  # (n_theta, n_r)

    ZZ = np.array(
        fd(jnp.array(TT.ravel()), jnp.array(RR.ravel()))
    ).reshape(TT.shape)

    # Cartesian coordinates for display
    XX = RR * np.cos(TT)
    YY = RR * np.sin(TT)

    if mode == "3d":
        fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5,
                                 figsize=(6.1, 2.58))
        facecolors = _matlab_facecolors(ZZ, cmap_obj)
        ax.plot_surface(
            XX,
            YY,
            ZZ,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
            **kw,
        )
        # Draw boundary circle at the base
        theta_bdy = np.linspace(0, 2 * np.pi, 300)
        zmin = float(ZZ.min())
        ax.plot(np.cos(theta_bdy), np.sin(theta_bdy),
                zs=zmin, zdir="z", color="k", linewidth=0.6, alpha=0.5)
        # MATLAB axis: exactly the unit square in x and y.
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
    else:
        # 2D flat mode
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.1, 2.58))
        else:
            fig = ax.get_figure()

        ax.pcolormesh(XX, YY, ZZ, cmap=cmap_obj, shading="auto", **kw)
        _draw_disk_boundary(ax, linewidth=0.8)
        ax.set_aspect("equal")
        _set_unit_ticks(ax, domain=(-1, 1, -1, 1))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    _apply_style(ax, title=title, grid=False)
    fig.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Sphere function plot (coloured sphere surface)
# ---------------------------------------------------------------------------

def plot_sphere(
    fs,
    ax=None,
    title: str = "",
    n_pts: int = 200,
    cmap=None,
    projection: str = "sphere",
    grid: bool = False,
    grid_line_type: str = "k-",
    n_grid_lam: int = 24,
    n_grid_th: int = 12,
    n_lam: int = None,  # backward-compat alias for n_pts
    n_theta: int = None,  # backward-compat (ignored; grid is uniform)
    **kw,
) -> tuple[plt.Figure, Any]:
    """Plot a Spherefun on the unit sphere (MATLAB Chebfun style).

    Faithful translation of @spherefun/surf.m from MATLAB Chebfun.

    Parameters
    ----------
    fs : Spherefun
    ax : Axes3D, optional
    title : str
    n_pts : int
        Grid resolution (default 200, matching MATLAB ``minPlotNum``).
    cmap : colormap, optional (default: parula)
    projection : str
        'sphere' (default), 'bumpy', 'equirectangular', 'hammer', 'albers',
        'eckert2', 'winkel3', 'sinusoidal'.
    grid : bool
        If True, overlay lat/lon grid lines.
    grid_line_type : str
        Matplotlib line spec for grid lines (default: 'k-').
    n_grid_lam, n_grid_th : int
        Number of grid lines in lon/lat directions.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    # Handle backward-compat aliases
    if n_lam is not None:
        n_pts = n_lam
    if n_theta is not None and n_lam is None:
        n_pts = n_theta

    # --- MATLAB: l = linspace(-pi, pi, 200); t = linspace(0, pi, 200) ---
    l = np.linspace(-np.pi, np.pi, n_pts)
    t = np.linspace(0.0, np.pi, n_pts)

    # --- MATLAB: C = fevalm(f, l, t) ---
    ll, tt = np.meshgrid(l, t)  # both (n_pts, n_pts)
    C = np.array(
        fs(jnp.array(ll.ravel()), jnp.array(tt.ravel()))
    ).reshape(ll.shape)

    # --- MATLAB: correction for near-constant functions ---
    if np.linalg.norm(C - C[0, 0], ord=np.inf) < 1e-10:
        C = np.full_like(C, C[0, 0])

    default_opts = dict(rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    # --- Grid meshes for lines of longitude/latitude ---
    llgl, ttgl = np.meshgrid(
        np.linspace(-np.pi, np.pi, n_grid_lam + 1),
        np.linspace(0, np.pi, n_pts))
    llgt, ttgt = np.meshgrid(
        np.linspace(-np.pi, np.pi, n_pts),
        np.linspace(0, np.pi, n_grid_th + 1))

    if projection.lower() in ('sphere', 'bumpy'):
        # 3D sphere plot
        vv = np.ones_like(ll)
        lim = [-1.0, 1.0]
        if projection.lower() == 'bumpy':
            scl = 0.15
            cmin, cmax = float(C.min()), float(C.max())
            if cmax > cmin:
                vv = vv + scl * (2.0 * (C - cmin) / (cmax - cmin) - 1.0)
            lim = [lim[0] - scl, lim[1] + scl]

        # MATLAB: [xx,yy,zz] = sph2cart(ll, pi/2-tt, vv)
        elev = np.pi / 2 - tt
        xx = vv * np.cos(elev) * np.cos(ll)
        yy = vv * np.cos(elev) * np.sin(ll)
        zz = vv * np.sin(elev)

        fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36,
                                 figsize=(6.1, 2.75), fill_canvas=False)

        facecolors = _matlab_facecolors(
            C,
            cmap_obj,
            shade_data=zz,
            apply_lighting=(projection.lower() == "bumpy"),
        )
        ax.plot_surface(xx, yy, zz, facecolors=facecolors, **default_opts, **kw)

        if grid:
            # Lines of longitude
            elg = np.pi / 2 - ttgl
            xxg = np.cos(elg) * np.cos(llgl)
            yyg = np.cos(elg) * np.sin(llgl)
            zzg = np.sin(elg)
            for i in range(xxg.shape[1]):
                ax.plot(xxg[:, i], yyg[:, i], zzg[:, i], grid_line_type, linewidth=0.5)
            # Lines of latitude
            elt = np.pi / 2 - ttgt
            xxt = np.cos(elt) * np.cos(llgt)
            yyt = np.cos(elt) * np.sin(llgt)
            zzt = np.sin(elt)
            for i in range(xxt.shape[0]):
                ax.plot(xxt[i, :], yyt[i, :], zzt[i, :], grid_line_type, linewidth=0.5)

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_box_aspect([1, 1, 1])

    else:
        # 2D map projections
        xh, yh = _sph2map(projection, ll, tt)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.1, 4.0))
        else:
            fig = ax.get_figure()

        ax.pcolormesh(xh, yh, C, cmap=cmap_obj, shading='auto', **kw)

        if grid:
            xg, yg = _sph2map(projection, llgl, ttgl)
            for i in range(xg.shape[1]):
                ax.plot(xg[:, i], yg[:, i], grid_line_type, linewidth=0.5)
            xg2, yg2 = _sph2map(projection, llgt.T, ttgt.T)
            for i in range(xg2.shape[1]):
                ax.plot(xg2[:, i], yg2[:, i], grid_line_type, linewidth=0.5)

        ax.set_aspect('equal')
        ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    return fig, ax


def _sph2map(projection: str, lam, th):
    """2D map projection from spherical coordinates (MATLAB @spherefun/surf.m sph2map).

    Parameters
    ----------
    projection : str
        One of 'equirectangular', 'hammer', 'albers', 'eckert2', 'winkel3', 'sinusoidal'.
    lam : ndarray
        Longitude in [-pi, pi].
    th : ndarray
        Colatitude in [0, pi].

    Returns
    -------
    xh, yh : ndarray
    """
    ptype = projection.lower()
    if ptype == 'equirectangular':
        return lam, th
    elif ptype == 'hammer':
        xh = 2.0 * np.sqrt(2) * np.sin(th) * np.sin(lam / 2) / np.sqrt(1 + np.sin(th) * np.cos(lam / 2))
        yh = np.sqrt(2) * np.cos(th) / np.sqrt(1 + np.sin(th) * np.cos(lam / 2))
        return xh, yh
    elif ptype == 'albers':
        th0 = np.pi / 2
        lam0 = 0.0
        th1 = np.pi / 2
        th2 = np.pi / 6
        n = 0.5 * (np.cos(th1) + np.cos(th2))
        phi = n * (lam - lam0)
        C = np.sin(th1) ** 2 + 2 * n * np.cos(th1) ** 2
        rho = np.sqrt(C - 2 * n * np.cos(th)) / n
        rho0 = np.sqrt(C - 2 * n * np.cos(th0)) / n
        xh = rho * np.sin(phi)
        yh = rho0 - rho * np.cos(phi)
        return xh, yh
    elif ptype == 'eckert2':
        lat = np.pi / 2 - th  # colatitude to latitude
        xh = 2 * lam * np.sqrt((4 - 3 * np.sin(np.abs(lat))) / (6 * np.pi))
        yh = np.sign(lat) * (np.sqrt(2 * np.pi / 3) * (2 - np.sqrt(4 - 3 * np.sin(np.abs(lat)))))
        return xh, yh
    elif ptype == 'winkel3':
        lat = np.pi / 2 - th
        th1 = np.arccos(2 / np.pi)
        alpha = np.arccos(np.cos(lat) * np.cos(lam / 2))
        sincalpha = np.where(alpha == 0, 1.0, np.sin(alpha) / alpha)
        xh = 0.5 * (lam * np.cos(th1) + 2 * np.cos(lat) * np.sin(lam / 2) / sincalpha)
        yh = 0.5 * (lat + np.sin(lat) / sincalpha)
        return xh, yh
    elif ptype == 'sinusoidal':
        lat = np.pi / 2 - th
        xh = lam * np.cos(lat)
        yh = lat
        return xh, yh
    else:
        # Default: equirectangular
        return lam, th


# ---------------------------------------------------------------------------
# Spherefun contour on the sphere
# ---------------------------------------------------------------------------


def contour_sphere(
    fs,
    ax=None,
    title: str = "",
    n_pts: int = 200,
    levels: int = 12,
    sphere_color=None,
    cmap=None,
    fmt=None,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Contour plot of a Spherefun on the unit sphere (MATLAB @spherefun/contour.m).

    Draws a near-white background sphere with contour lines in 3D.

    Parameters
    ----------
    fs : Spherefun
    ax : Axes3D, optional
    title : str
    n_pts : int
    levels : int or array-like
    sphere_color : tuple, optional (default: off-white)
    cmap : colormap, optional (default: parula)

    Returns
    -------
    fig, ax
    """
    # MATLAB convention: contour(f, [v v]) draws the single level v.
    if not np.isscalar(levels):
        lv = np.atleast_1d(np.asarray(levels, dtype=float))
        if lv.size == 2 and lv[0] == lv[1]:
            levels = [float(lv[0])]
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    if sphere_color is None:
        sphere_color = (250 / 255, 250 / 255, 250 / 255)

    # Evaluate on grid
    l = np.linspace(-np.pi, np.pi, n_pts)
    t = np.linspace(0.0, np.pi, n_pts)
    ll, tt = np.meshgrid(l, t)
    C = np.array(
        fs(jnp.array(ll.ravel()), jnp.array(tt.ravel()))
    ).reshape(ll.shape)

    # Get contour lines using a temporary 2D contour call
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(l, t, C, levels=levels)

    level_list = cs.levels
    clrmap = cmap_obj(np.linspace(0, 1, max(len(level_list), 1)))

    # Extract contour paths — compatible with both old and new matplotlib
    contour_paths = []
    if hasattr(cs, 'allsegs'):
        # Matplotlib >= 3.8: use allsegs/allkinds
        for i, segs in enumerate(cs.allsegs):
            lev = level_list[i] if i < len(level_list) else level_list[-1]
            for seg in segs:
                if len(seg) > 1:
                    contour_paths.append((lev, seg))
    elif hasattr(cs, 'collections'):
        # Matplotlib < 3.8: use collections
        for i, collection in enumerate(cs.collections):
            lev = level_list[i] if i < len(level_list) else level_list[-1]
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) > 1:
                    contour_paths.append((lev, vertices))
    else:
        # Fallback: extract from contour matrix C
        pass
    plt.close(fig_tmp)

    # Setup 3D axes
    fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36, figsize=(6.1, 2.75))

    _draw_sphere_background(ax, color=sphere_color)

    # Plot contour lines on sphere
    for lev_val, verts in contour_paths:
        lam_c = verts[:, 0]
        th_c = verts[:, 1]
        # Colatitude to Cartesian on sphere
        xv = np.sin(th_c) * np.cos(lam_c)
        yv = np.sin(th_c) * np.sin(lam_c)
        zv = np.cos(th_c)
        # Color from level (or a fixed linespec colour when given)
        if fmt is not None:
            clr = next((ch for ch in fmt if ch in "bgrcmykw"), "k")
            lstyle = next((ls for ls in ("--", "-.", ":", "-")
                           if ls in fmt), "-")
            ax.plot(xv, yv, zv, color=clr, linestyle=lstyle, linewidth=1.0)
            continue
        if len(level_list) > 1:
            idx = np.argmin(np.abs(lev_val - level_list))
            clr = clrmap[idx, :3]
        else:
            clr = 'k'
        ax.plot(xv, yv, zv, color=clr, linewidth=1.0)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Spherefun quiver (3 Cartesian components) — @spherefun/quiver.m
# ---------------------------------------------------------------------------


def quiver_sphere_cartesian(
    fx, fy, fz,
    ax=None,
    title: str = "",
    n_refine: int = 4,
    sphere_color=None,
    arrow_scale: float = 2.0,
    arrow_color: str = "k",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Quiver plot of a vector field given by three Spherefuns in Cartesian coords.

    Faithful translation of @spherefun/quiver.m. Arrows are placed at
    icosahedral nodes for well-separated coverage.

    Parameters
    ----------
    fx, fy, fz : Spherefun
        Three Cartesian components of the vector field.
    ax : Axes3D, optional
    title : str
    n_refine : int
        Icosahedral refinement level (4 gives 2562 nodes).
    sphere_color : tuple, optional
    arrow_scale : float
    arrow_color : str

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    if sphere_color is None:
        sphere_color = (255 / 255, 255 / 255, 204 / 255)

    # Generate icosahedral nodes
    nodes = _icos_nodes(n_refine)

    # Convert Cartesian nodes to spherical (lam, theta) for Spherefun evaluation
    x_n, y_n, z_n = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    lam_n = np.arctan2(y_n, x_n)
    theta_n = np.arccos(np.clip(z_n, -1, 1))

    lam_j = jnp.array(lam_n)
    theta_j = jnp.array(theta_n)

    fxv = np.array(fx(lam_j, theta_j))
    fyv = np.array(fy(lam_j, theta_j))
    fzv = np.array(fz(lam_j, theta_j))

    fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36, figsize=(6.1, 2.75))

    _draw_sphere_background(ax, color=sphere_color)

    mag = np.sqrt(fxv ** 2 + fyv ** 2 + fzv ** 2)
    max_mag = float(mag.max()) if mag.size and float(mag.max()) > 0 else 1.0
    scale = 0.08 * arrow_scale / max_mag
    ax.quiver(
        x_n,
        y_n,
        z_n,
        fxv * scale,
        fyv * scale,
        fzv * scale,
        length=1.0,
        color=arrow_color,
        arrow_length_ratio=0.3,
        linewidth=0.8,
        **kw,
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


def _icos_nodes(k: int = 4) -> np.ndarray:
    """Generate icosahedral nodes on the unit sphere by *k* levels of bisection.

    Faithful translation of getIcosNodes(k, 0) from @spherefunv/quiver.m.

    Returns
    -------
    x : ndarray, shape (N, 3)
        Cartesian coordinates of the nodes.
    """
    p = (1 + np.sqrt(5)) / 2
    x = np.array([
        [0, p, 1], [0, -p, 1], [0, p, -1], [0, -p, -1],
        [1, 0, p], [-1, 0, p], [1, 0, -p], [-1, 0, -p],
        [p, 1, 0], [-p, 1, 0], [p, -1, 0], [-p, -1, 0],
    ], dtype=float)
    # Normalize to unit sphere
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    # Simple triangulation from convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(x)
    tri = hull.simplices

    # Bisect k times
    for _ in range(k):
        x, tri = _bisect_tri(x, tri)

    return x


def _bisect_tri(x: np.ndarray, tri: np.ndarray):
    """Bisect each triangle in mesh (x, tri) and project to sphere."""
    Nx = len(x)
    Nt = len(tri)

    v1 = (x[tri[:, 0]] + x[tri[:, 1]]) / 2
    v2 = (x[tri[:, 1]] + x[tri[:, 2]]) / 2
    v3 = (x[tri[:, 2]] + x[tri[:, 0]]) / 2
    v = np.vstack([v1, v2, v3])

    # Remove duplicates
    v_unique, idx = np.unique(np.round(v, 12), axis=0, return_inverse=True)

    i1 = Nx + idx[:Nt]
    i2 = Nx + idx[Nt:2 * Nt]
    i3 = Nx + idx[2 * Nt:]

    t1 = np.column_stack([tri[:, 0], i1, i3])
    t2 = np.column_stack([tri[:, 1], i2, i1])
    t3 = np.column_stack([tri[:, 2], i3, i2])
    t4 = np.column_stack([i1, i2, i3])

    x_new = np.vstack([x, v_unique])
    x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
    tri_new = np.vstack([t1, t2, t3, t4])

    return x_new, tri_new


# ---------------------------------------------------------------------------
# Chebfun2 quiver — @chebfun2/quiver.m (delegates to separableApprox)
# ---------------------------------------------------------------------------


def quiver_2d(
    f2,
    g2=None,
    ax=None,
    title: str = "",
    n_pts: int = 10,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """2D quiver plot of a Chebfun2 gradient or a Chebfun2v vector field.

    If *f2* is a Chebfun2v (or two Chebfun2 components are given as f2, g2),
    plots the velocity field (f2, g2) using matplotlib quiver.

    Faithful translation of @separableApprox/quiver.m and @chebfun2v/quiver.m.

    Parameters
    ----------
    f2 : Chebfun2v, or first Chebfun2 component
    g2 : Chebfun2, optional
        Second component (if f2 is a Chebfun2).
    ax : Axes, optional
    title : str
    n_pts : int
        Number of arrows per axis direction.

    Returns
    -------
    fig, ax
    """

    # Determine components
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    if isinstance(f2, Chebfun2v):
        F1, F2 = f2.components[0], f2.components[1]
        try:
            x0, x1, y0, y1 = f2.domain
        except Exception:
            x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0
    else:
        if g2 is None:
            raise ValueError("quiver_2d requires either a Chebfun2v or two Chebfun2 arguments.")
        F1, F2 = f2, g2
        try:
            x0, x1, y0, y1 = F1.domain
        except Exception:
            x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    UU = _eval_2d_vectorized(F1, XX, YY)
    VV = _eval_2d_vectorized(F2, XX, YY)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.1, 2.75))
    else:
        fig = ax.get_figure()

    ax.quiver(XX, YY, UU, VV, **kw)
    ax.set_xlim(float(x0) * 1.1, float(x1) * 1.1)
    ax.set_ylim(float(y0) * 1.1, float(y1) * 1.1)
    ax.set_aspect('equal')
    _apply_style(ax, title=title)
    fig.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Chebfun2v parametric surface — @chebfun2v/surf.m
# ---------------------------------------------------------------------------


def surf_chebfun2v(
    fv,
    ax=None,
    title: str = "",
    n_pts: int = 100,
    cmap=None,
    show_seams: bool = False,
    seam_color: str = "k",
    seam_linestyle: str = "-",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Surface plot of a 3-component Chebfun2v as a parametric surface.

    Faithful translation of @chebfun2v/surf.m. The three components
    define x(u,v), y(u,v), z(u,v) where (u,v) ranges over the domain.

    Parameters
    ----------
    fv : Chebfun2v
        Must have 3 components.
    ax : Axes3D, optional
    title : str
    n_pts : int
    cmap : colormap, optional
    show_seams : bool
        If True, overlay boundary seam lines.
    seam_color, seam_linestyle : str

    Returns
    -------
    fig, ax
    """
    if len(fv.components) < 3:
        raise ValueError("surf_chebfun2v requires a Chebfun2v with 3 components.")

    if cmap is None:
        cmap = PARULA

    F1, F2, F3 = fv.components[0], fv.components[1], fv.components[2]
    try:
        x0, x1, y0, y1 = fv.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    Xs = _eval_2d_vectorized(F1, XX, YY)
    Ys = _eval_2d_vectorized(F2, XX, YY)
    Zs = _eval_2d_vectorized(F3, XX, YY)

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5, figsize=(6.1, 2.75))

    ax.plot_surface(Xs, Ys, Zs, cmap=cmap,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=True, **kw)

    if show_seams:
        import jax.numpy as jnp
        LW = 2
        # Bottom seam: y = y0
        xpts = np.linspace(float(x0), float(x1), n_pts)
        lft_y = np.full_like(xpts, float(y0))
        x1v = np.array(F1(jnp.array(xpts), jnp.array(lft_y)))
        y1v = np.array(F2(jnp.array(xpts), jnp.array(lft_y)))
        z1v = np.array(F3(jnp.array(xpts), jnp.array(lft_y)))
        ax.plot(x1v, y1v, z1v, linestyle=seam_linestyle, color=seam_color, linewidth=LW)
        # Top seam: y = y1
        rght_y = np.full_like(xpts, float(y1))
        x2v = np.array(F1(jnp.array(xpts), jnp.array(rght_y)))
        y2v = np.array(F2(jnp.array(xpts), jnp.array(rght_y)))
        z2v = np.array(F3(jnp.array(xpts), jnp.array(rght_y)))
        ax.plot(x2v, y2v, z2v, linestyle=seam_linestyle, color=seam_color, linewidth=LW)
        # Left seam: x = x0
        ypts = np.linspace(float(y0), float(y1), n_pts)
        dwn_x = np.full_like(ypts, float(x0))
        x3v = np.array(F1(jnp.array(dwn_x), jnp.array(ypts)))
        y3v = np.array(F2(jnp.array(dwn_x), jnp.array(ypts)))
        z3v = np.array(F3(jnp.array(dwn_x), jnp.array(ypts)))
        ax.plot(x3v, y3v, z3v, linestyle=seam_linestyle, color=seam_color, linewidth=LW)
        # Right seam: x = x1
        up_x = np.full_like(ypts, float(x1))
        x4v = np.array(F1(jnp.array(up_x), jnp.array(ypts)))
        y4v = np.array(F2(jnp.array(up_x), jnp.array(ypts)))
        z4v = np.array(F3(jnp.array(up_x), jnp.array(ypts)))
        ax.plot(x4v, y4v, z4v, linestyle=seam_linestyle, color=seam_color, linewidth=LW)

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Disk contour — @diskfun/contour.m
# ---------------------------------------------------------------------------


def contour_disk(
    fd,
    ax=None,
    title: str = "",
    n_pts: int = 200,
    levels: int = 12,
    cmap=None,
    fmt=None,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Contour plot of a Diskfun on the unit disk (MATLAB @diskfun/contour.m).

    Evaluates on a polar grid, converts to Cartesian, and overlays a
    boundary circle.

    Parameters
    ----------
    fd : Diskfun
    ax : Axes, optional
    title : str
    n_pts : int
    levels : int or array-like
    cmap : colormap, optional

    Returns
    -------
    fig, ax
    """
    # MATLAB convention: contour(f, [v v]) draws the single level v.
    if not np.isscalar(levels):
        lv = np.atleast_1d(np.asarray(levels, dtype=float))
        if lv.size == 2 and lv[0] == lv[1]:
            levels = [float(lv[0])]
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    # Evaluate on polar grid
    theta = np.linspace(-np.pi, np.pi, n_pts)
    r = np.linspace(0.0, 1.0, n_pts)
    TT, RR = np.meshgrid(theta, r)
    vals = np.array(
        fd(jnp.array(TT.ravel()), jnp.array(RR.ravel()))
    ).reshape(TT.shape)

    # Convert to Cartesian
    XX = RR * np.cos(TT)
    YY = RR * np.sin(TT)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.1, 2.75))
    else:
        fig = ax.get_figure()

    if fmt is not None:
        colors = next((ch for ch in fmt if ch in "bgrcmykw"), "k")
        lstyle = next((ls for ls in ("--", "-.", ":", "-") if ls in fmt),
                      "-")
        ax.contour(XX, YY, vals, levels=levels, colors=colors,
                   linestyles=lstyle, **kw)
    else:
        ax.contour(XX, YY, vals, levels=levels, cmap=cmap_obj, **kw)

    _draw_disk_boundary(ax, linewidth=0.3)

    ax.set_aspect('equal')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    _apply_style(ax, title=title, grid=False)
    fig.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Disk surface — @diskfun/surf.m
# ---------------------------------------------------------------------------


def surf_disk(
    fd,
    ax=None,
    title: str = "",
    n_pts: int = 200,
    cmap=None,
    **kw,
) -> tuple[plt.Figure, Any]:
    """3D surface plot of a Diskfun on the unit disk (MATLAB @diskfun/surf.m).

    Evaluates f on a polar grid (theta, r), converts to Cartesian (x, y),
    and plots f values as z-height.

    Parameters
    ----------
    fd : Diskfun
    ax : Axes3D, optional
    title : str
    n_pts : int
    cmap : colormap, optional

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    theta = np.linspace(-np.pi, np.pi, n_pts)
    r = np.linspace(0.0, 1.0, n_pts)
    TT, RR = np.meshgrid(theta, r)

    C = np.array(
        fd(jnp.array(TT.ravel()), jnp.array(RR.ravel()))
    ).reshape(TT.shape)

    # Correction for near-constant
    if np.linalg.norm(C - C[0, 0], ord=np.inf) < 1e-10:
        C = np.full_like(C, C[0, 0])

    xx = RR * np.cos(TT)
    yy = RR * np.sin(TT)

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5, figsize=(6.1, 2.75))

    facecolors = _matlab_facecolors(C, cmap_obj)
    ax.plot_surface(
        xx,
        yy,
        C,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
        **kw,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Diskfunv quiver — @diskfunv/quiver.m
# ---------------------------------------------------------------------------


def quiver_disk(
    fv,
    ax=None,
    title: str = "",
    n_pts: int = 30,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Quiver plot of a Diskfunv on the unit disk (MATLAB @diskfunv/quiver.m).

    Arrows are placed at approximately equally-spaced points inside the
    unit disk, matching the MATLAB diskpts() algorithm.

    Parameters
    ----------
    fv : Diskfunv
    ax : Axes, optional
    title : str
    n_pts : int
        Approximate number of arrows (controls density).

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    # Generate disk sampling points (matching MATLAB diskpts)
    xx, yy = _disk_pts(n_pts)

    # Convert to polar for evaluation
    rr = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    F1, F2 = fv.components
    # Evaluate in Cartesian (diskfunv components are in x,y coords)
    vals1 = np.array(F1(jnp.array(theta), jnp.array(rr)))
    vals2 = np.array(F2(jnp.array(theta), jnp.array(rr)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.1, 2.75))
    else:
        fig = ax.get_figure()

    _draw_disk_boundary(ax, dashed=True, linewidth=0.5)

    ax.quiver(xx, yy, vals1, vals2, **kw)
    ax.set_aspect('equal')
    mx = max(np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_ylim())), 1.0)
    ax.set_xlim(-mx * 1.1, mx * 1.1)
    ax.set_ylim(-mx * 1.1, mx * 1.1)

    _apply_style(ax, title=title, grid=False)
    fig.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    return fig, ax


def _disk_pts(n_pts: int):
    """Generate approximately equally-spaced points on the unit disk.

    Faithful translation of the diskpts() function from @diskfunv/quiver.m.
    Uses concentric rings with increasing angular density.

    Parameters
    ----------
    n_pts : int
        Target number of points.

    Returns
    -------
    xx, yy : ndarray
    """
    n = max(int(np.floor(n_pts / np.sqrt(3))), 1)
    dr = 1.0 / n

    xx_list = [0.0]
    yy_list = [0.0]

    # Second ring: 6 points
    th = np.linspace(-np.pi, np.pi, 7)[:-1]  # 6 equally spaced
    xx_list.extend((dr * np.cos(th)).tolist())
    yy_list.extend((dr * np.sin(th)).tolist())

    # Subsequent rings: 3*(2*k - 1) points
    for k in range(2, n + 1):
        nk = 3 * (2 * k - 1)
        th = np.linspace(-np.pi, np.pi, nk + 1)[:-1]
        rk = dr * k
        xx_list.extend((rk * np.cos(th)).tolist())
        yy_list.extend((rk * np.sin(th)).tolist())

    return np.array(xx_list), np.array(yy_list)


# ---------------------------------------------------------------------------
# Chebfun3 slice plots
# ---------------------------------------------------------------------------

def plot_slices(
    f3,
    ax=None,
    title: str = "",
    n_pts: int = 80,
    cmap=None,
    alpha: float = 0.85,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Three orthogonal mid-plane slices of a Chebfun3 (MATLAB Chebfun style).

    Plots the z=mid, y=mid, and x=mid slices as filled colour images on
    3-D axes with parula colormap and consistent colour limits.

    Parameters
    ----------
    f3 : Chebfun3
    ax : Axes3D, optional
    title : str
    n_pts : int
    cmap : colormap, optional (default: parula)
    alpha : float
        Surface transparency.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp
    import matplotlib.colors as mcolors

    if cmap is None:
        cmap = PARULA

    try:
        xa, xb, ya, yb, za, zb = f3.domain
    except Exception:
        xa, xb, ya, yb, za, zb = -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

    xa, xb = float(xa), float(xb)
    ya, yb = float(ya), float(yb)
    za, zb = float(za), float(zb)

    xm = 0.5 * (xa + xb)
    ym = 0.5 * (ya + yb)
    zm = 0.5 * (za + zb)

    xs = np.linspace(xa, xb, n_pts)
    ys = np.linspace(ya, yb, n_pts)
    zs = np.linspace(za, zb, n_pts)

    # --- z = zm slice (XY plane) ---
    XX_xy, YY_xy = np.meshgrid(xs, ys, indexing="ij")
    ZM_xy = np.full_like(XX_xy, zm)
    F_xy = np.array(
        f3(jnp.array(XX_xy.ravel()), jnp.array(YY_xy.ravel()),
           jnp.array(ZM_xy.ravel()))
    ).reshape(XX_xy.shape)

    # --- y = ym slice (XZ plane) ---
    XX_xz, ZZ_xz = np.meshgrid(xs, zs, indexing="ij")
    YM_xz = np.full_like(XX_xz, ym)
    F_xz = np.array(
        f3(jnp.array(XX_xz.ravel()), jnp.array(YM_xz.ravel()),
           jnp.array(ZZ_xz.ravel()))
    ).reshape(XX_xz.shape)

    # --- x = xm slice (YZ plane) ---
    YY_yz, ZZ_yz = np.meshgrid(ys, zs, indexing="ij")
    XM_yz = np.full_like(YY_yz, xm)
    F_yz = np.array(
        f3(jnp.array(XM_yz.ravel()), jnp.array(YY_yz.ravel()),
           jnp.array(ZZ_yz.ravel()))
    ).reshape(YY_yz.shape)

    # Global colour limits
    all_vals = np.concatenate([F_xy.ravel(), F_xz.ravel(), F_yz.ravel()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5,
                             figsize=(6.1, 2.58))

    def _surf_slice(XX, YY, ZZ, F):
        fc = cmap_obj(norm(F))
        ax.plot_surface(XX, YY, ZZ, facecolors=fc,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True,
                        alpha=alpha, shade=False)

    _surf_slice(XX_xy, YY_xy, ZM_xy, F_xy)
    _surf_slice(XX_xz, YM_xz, ZZ_xz, F_xz)
    _surf_slice(XM_yz, YY_yz, ZZ_yz, F_yz)

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Spherefunv quiver plot (vector field on sphere)
# ---------------------------------------------------------------------------


def quiver_sphere(
    fv,
    ax=None,
    title: str = "",
    n_lam: int = 20,
    n_theta: int = 10,
    sphere_color=(255 / 255, 255 / 255, 204 / 255),
    arrow_color: str = "k",
    arrow_scale: float = 1.0,
    cmap=None,
    use_icos: bool = True,
    n_refine: int = 4,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Quiver plot of a Spherefunv on the unit sphere (MATLAB Chebfun style).

    Faithful translation of @spherefunv/quiver.m. By default uses a lat/lon
    grid; with ``use_icos=True``, uses icosahedral nodes (matching MATLAB).

    The Spherefunv components are internally converted to Cartesian arrows
    on the sphere.

    For a 3-component Spherefunv (Cartesian components), the arrows are
    plotted directly. For a 2-component Spherefunv (tangent components),
    arrows are converted via the standard tangent-vector basis.

    Parameters
    ----------
    fv : Spherefunv
        Vector field on the sphere.
    ax : Axes3D, optional
    title : str
    n_lam, n_theta : int
        Sampling density when using a regular grid (use_icos=False).
    sphere_color : str
        Background sphere surface colour.
    arrow_color : str
        Arrow colour.
    arrow_scale : float
        Scale factor for arrow length.
    cmap : colormap, optional
    use_icos : bool
        If True, use icosahedral nodes (matching @spherefunv/quiver.m).
    n_refine : int
        Icosahedral refinement level (only used when use_icos=True).

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    n_comps = len(fv.components)

    if use_icos:
        nodes = _icos_nodes(n_refine)
        x_n, y_n, z_n = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        lam_n = jnp.array(np.arctan2(y_n, x_n))
        theta_n = jnp.array(np.arccos(np.clip(z_n, -1, 1)))

        if n_comps == 3:
            fxv = np.array(fv.components[0](lam_n, theta_n))
            fyv = np.array(fv.components[1](lam_n, theta_n))
            fzv = np.array(fv.components[2](lam_n, theta_n))
        else:
            # 2-component: convert tangent to Cartesian
            f_vals = np.array(fv.components[0](lam_n, theta_n))
            g_vals = np.array(fv.components[1](lam_n, theta_n))
            sin_th = np.sin(np.array(theta_n))
            cos_th = np.cos(np.array(theta_n))
            sin_lam = np.sin(np.array(lam_n))
            cos_lam = np.cos(np.array(lam_n))
            fxv = f_vals * (-sin_lam) + g_vals * cos_th * cos_lam
            fyv = f_vals * cos_lam + g_vals * cos_th * sin_lam
            fzv = -g_vals * sin_th

        fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36, figsize=(6.1, 2.75))

        _draw_sphere_background(ax, color=sphere_color)

        mag = np.sqrt(fxv ** 2 + fyv ** 2 + fzv ** 2)
        max_mag = float(mag.max()) if mag.size and float(mag.max()) > 0 else 1.0
        scale = 0.08 * arrow_scale / max_mag
        ax.quiver(
            x_n,
            y_n,
            z_n,
            fxv * scale,
            fyv * scale,
            fzv * scale,
            length=1.0,
            color=arrow_color,
            arrow_length_ratio=0.25,
            linewidth=0.8,
            **kw,
        )

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect([1, 1, 1])

    else:
        # Original lat/lon grid path for 2-component tangent vector field
        lam = np.linspace(-np.pi, np.pi, n_lam, endpoint=False)
        theta = np.linspace(0.15, np.pi - 0.15, n_theta)
        LAM, THETA = np.meshgrid(lam, theta, indexing="ij")

        lam_flat = jnp.array(LAM.ravel())
        theta_flat = jnp.array(THETA.ravel())

        f_comp, g_comp = fv.components
        f_vals = np.array(f_comp(lam_flat, theta_flat)).reshape(LAM.shape)
        g_vals = np.array(g_comp(lam_flat, theta_flat)).reshape(LAM.shape)

        X = np.sin(THETA) * np.cos(LAM)
        Y = np.sin(THETA) * np.sin(LAM)
        Z = np.cos(THETA)

        sin_th = np.sin(THETA)
        cos_th = np.cos(THETA)
        sin_lam = np.sin(LAM)
        cos_lam = np.cos(LAM)

        U = f_vals * (-sin_lam) + g_vals * cos_th * cos_lam
        V = f_vals * cos_lam + g_vals * cos_th * sin_lam
        W = -g_vals * sin_th

        mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)
        max_mag = float(mag.max()) if mag.size and float(mag.max()) > 0 else 1.0
        scale = 0.08 * arrow_scale / max_mag
        U = U * scale
        V = V * scale
        W = W * scale

        fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36, figsize=(6.1, 2.75))
        _draw_sphere_background(ax, color=sphere_color)

        ax.quiver(X.ravel(), Y.ravel(), Z.ravel(),
                  U.ravel(), V.ravel(), W.ravel(),
                  color=arrow_color, arrow_length_ratio=0.25,
                  linewidth=0.8, length=1.0, **kw)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Ballfun isosurface plot
# ---------------------------------------------------------------------------


def isosurface_ball(
    bf,
    levels=None,
    ax=None,
    title: str = "",
    n_pts: int = 50,
    cmap=None,
    alpha: float = 0.6,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Isosurface plot of a Ballfun (level sets inside the unit ball).

    Uses the marching cubes algorithm to extract isosurfaces and renders
    them as 3D polygon collections.

    Parameters
    ----------
    bf : Ballfun
        The 3-D function on the ball.
    levels : list of float, optional
        Isosurface level values.  Defaults to 3 levels spanning the range.
    ax : Axes3D, optional
    title : str
    n_pts : int
        Grid resolution for marching cubes.
    cmap : colormap, optional (default: parula)
    alpha : float
        Surface transparency.

    Returns
    -------
    fig, ax
    """
    import jax
    import jax.numpy as jnp

    if cmap is None:
        cmap = PARULA

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Build a Cartesian grid inside the ball
    t = np.linspace(-1.0, 1.0, n_pts)
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Convert to spherical
    LAM = np.arctan2(Y, X)
    THETA = np.where(R > 0,
                     np.arccos(np.clip(Z / np.maximum(R, 1e-16), -1, 1)),
                     0.0)

    # Evaluate Ballfun on the grid (only inside the ball)
    mask = R <= 1.0
    vals = np.full(R.shape, np.nan)
    idx = mask.ravel()
    if idx.any():
        r_pts = jnp.array(R.ravel()[idx])
        l_pts = jnp.array(LAM.ravel()[idx])
        t_pts = jnp.array(THETA.ravel()[idx])
        eval_fn = jax.vmap(lambda ri, li, ti: bf(ri, li, ti))
        v = np.asarray(eval_fn(r_pts, l_pts, t_pts))
        vals.ravel()[idx] = v

    # Replace NaN outside ball with boundary value for marching cubes
    vmin_real = float(np.nanmin(vals))
    vmax_real = float(np.nanmax(vals))
    vals = np.where(np.isnan(vals), vmin_real - 1.0, vals)

    # Determine isosurface levels
    if levels is None:
        levels = np.linspace(vmin_real + 0.1 * (vmax_real - vmin_real),
                             vmax_real - 0.1 * (vmax_real - vmin_real), 3)

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5,
                             figsize=(6.1, 2.58))

    try:
        from skimage.measure import marching_cubes
        _have_skimage = True
    except ImportError:
        _have_skimage = False

    if _have_skimage:
        import matplotlib.colors as mcolors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        norm = mcolors.Normalize(vmin=min(levels), vmax=max(levels))

        for i, lev in enumerate(levels):
            try:
                verts, faces, _, _ = marching_cubes(vals, level=lev,
                                                     spacing=(2.0 / (n_pts - 1),) * 3)
                # Shift from grid indices to physical coords [-1, 1]
                verts = verts - 1.0  # marching_cubes returns in spacing units

                mesh = Poly3DCollection(verts[faces], alpha=alpha,
                                        linewidth=0)
                color = cmap_obj(norm(lev))
                mesh.set_facecolor(color)
                mesh.set_edgecolor((*color[:3], 0.1))
                ax.add_collection3d(mesh)
            except Exception:
                pass  # skip levels with no surface
    else:
        # Fallback: plot a single contour slice through z=0
        import warnings
        warnings.warn("scikit-image not found; falling back to mid-plane slice")
        mid = n_pts // 2
        ax.contourf(X[:, :, mid], Y[:, :, mid], vals[:, :, mid],
                     levels=20, cmap=cmap_obj, alpha=0.8)

    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.08, linewidth=0.3)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.1, 1.1)

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Ballfun 3-orthogonal-slice plot (MATLAB Chebfun style)
# ---------------------------------------------------------------------------


def plot_ball_slices(
    bf,
    ax=None,
    title: str = "",
    cmap=None,
    elev: float = 30,
    azim: float = -37.5,
    style: str = "ball",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Slice plot of a Ballfun inside the unit ball (MATLAB Chebfun style).

    Faithful translation of ``plotBall`` from @ballfun/plot.m.
    Produces 5 surfaces: 1 sphere at r≈0.5, 2 constant-elevation slices,
    2 constant-lambda half-planes. Coordinates use MATLAB's sph2cart
    convention where theta is elevation [-pi/2, pi/2].

    Parameters
    ----------
    bf : Ballfun
        The 3-D function on the unit ball.
    ax : Axes3D, optional
    title : str
    cmap : colormap, optional (default: parula)
    elev, azim : float
        Camera view angles. Default is MATLAB's view(3): elev=30, azim=-37.5.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    m, n, p = map(int, bf.shape)
    m = max(25, m)
    n = max(28, n)
    p = max(28, p)
    m = m + ((1 - (m % 6)) % 6)
    n = n + ((4 - (n % 4)) % 4)
    p = p + ((4 - (p % 4)) % 4)

    r_full = np.asarray(chebpts(m))
    r_pos = r_full[m // 2 :]

    if ax is None:
        fig = plt.figure(figsize=(6.1, 2.75))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    def _sph2cart(az, el, radius):
        return (
            radius * np.cos(el) * np.cos(az),
            radius * np.cos(el) * np.sin(az),
            radius * np.sin(el),
        )

    def _ball_cartesian(lam_vals, th_colat_vals, radius_vals):
        xs = radius_vals * np.sin(th_colat_vals) * np.cos(lam_vals)
        ys = radius_vals * np.sin(th_colat_vals) * np.sin(lam_vals)
        zs = radius_vals * np.cos(th_colat_vals)
        xs, ys, zs = np.broadcast_arrays(xs, ys, zs)
        return xs, ys, zs

    surfaces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    style_name = style.lower()

    if style_name == "ball":
        lam_core = np.asarray(trigpts(n)[0]) * np.pi
        lam = np.concatenate([lam_core, [np.pi]])
        th = np.concatenate([np.asarray(trigpts(p)[0]) * np.pi, [np.pi]]) - np.pi / 2.0
        th = th[p // 2 :]
        th_colat = np.pi / 2.0 - th

        ff = np.asarray(bf.fevalm(jnp.asarray(r_pos), jnp.asarray(lam), jnp.asarray(th_colat)))
        if bf.is_real:
            ff = np.real(ff)

        idx_r = int(np.argmin(np.abs(r_pos - 0.5)))
        tslice = [float(th[0]), float(th[p // 4])]
        lslice = [float(lam[0]), float(lam[n // 4])]

        for th_val in tslice:
            th_idx = int(np.argmin(np.abs(th - th_val)))
            cdata = ff[:, :, th_idx]
            rr, ll = np.meshgrid(r_pos, lam, indexing="ij")
            xs, ys, zs = _sph2cart(ll, th_val, rr)
            surfaces.append((xs, ys, zs, cdata))

        cdata = ff[idx_r, :, :]
        ll, tt = np.meshgrid(lam, th, indexing="ij")
        xs, ys, zs = _sph2cart(ll, tt, float(r_pos[idx_r]))
        surfaces.append((xs, ys, zs, cdata))

        for lam_val in lslice:
            lam_idx = int(np.argmin(np.abs(lam - lam_val)))
            cdata = ff[:, lam_idx, :]
            rr, tt = np.meshgrid(r_pos, th, indexing="ij")
            xs, ys, zs = _sph2cart(lam_val, tt, rr)
            surfaces.append((xs, ys, zs, cdata))

    elif style_name == "wedgeaz":
        az_intvl = (-np.pi / 2.0, np.pi)
        lam = np.linspace(az_intvl[0], az_intvl[1], n)
        th_colat = np.linspace(0.0, np.pi, p)

        ff = np.asarray(bf.fevalm(jnp.asarray([1.0]), jnp.asarray(lam), jnp.asarray(th_colat)))[0]
        if bf.is_real:
            ff = np.real(ff)
        xs, ys, zs = _ball_cartesian(lam[None, :], th_colat[:, None], 1.0)
        surfaces.append((xs, ys, zs, ff.T))

        for lam_val in (lam[0], lam[-1]):
            ff = np.asarray(
                bf.fevalm(jnp.asarray(r_pos), jnp.asarray([lam_val]), jnp.asarray(th_colat))
            )[:, 0, :]
            if bf.is_real:
                ff = np.real(ff)
            rr, tt = np.meshgrid(r_pos, th_colat, indexing="ij")
            xs, ys, zs = _ball_cartesian(lam_val, tt, rr)
            surfaces.append((xs, ys, zs, ff))

    elif style_name == "wedgepol":
        pol_intvl = (np.pi / 2.0, np.pi)
        az_intvl = (0.0, np.pi)

        lam_full = np.linspace(-np.pi, np.pi, n)
        th_cap = np.linspace(pol_intvl[0], pol_intvl[1], p)
        ff = np.asarray(bf.fevalm(jnp.asarray([1.0]), jnp.asarray(lam_full), jnp.asarray(th_cap)))[0]
        if bf.is_real:
            ff = np.real(ff)
        xs, ys, zs = _ball_cartesian(lam_full[None, :], th_cap[:, None], 1.0)
        surfaces.append((xs, ys, zs, ff.T))

        lam_open = np.linspace(az_intvl[0], az_intvl[1], n)
        th_open = np.linspace(0.0, pol_intvl[0], p)
        ff = np.asarray(bf.fevalm(jnp.asarray([1.0]), jnp.asarray(lam_open), jnp.asarray(th_open)))[0]
        if bf.is_real:
            ff = np.real(ff)
        xs, ys, zs = _ball_cartesian(lam_open[None, :], th_open[:, None], 1.0)
        surfaces.append((xs, ys, zs, ff.T))

        for lam_val in (lam_open[0], lam_open[-1]):
            ff = np.asarray(
                bf.fevalm(jnp.asarray(r_pos), jnp.asarray([lam_val]), jnp.asarray(th_open))
            )[:, 0, :]
            if bf.is_real:
                ff = np.real(ff)
            rr, tt = np.meshgrid(r_pos, th_open, indexing="ij")
            xs, ys, zs = _ball_cartesian(lam_val, tt, rr)
            surfaces.append((xs, ys, zs, ff))

        lam_eq = np.linspace(-np.pi, np.pi, p)
        ff = np.asarray(
            bf.fevalm(jnp.asarray(r_pos), jnp.asarray(lam_eq), jnp.asarray([pol_intvl[0]]))
        )[:, :, 0]
        if bf.is_real:
            ff = np.real(ff)
        rr, ll = np.meshgrid(r_pos, lam_eq, indexing="ij")
        xs, ys, zs = _ball_cartesian(ll, pol_intvl[0], rr)
        surfaces.append((xs, ys, zs, ff))

    else:
        raise ValueError(f"Unknown Ballfun plot style {style!r}. Expected 'ball', 'wedgeaz', or 'wedgepol'.")

    all_values = np.concatenate([surface[3].ravel() for surface in surfaces])
    norm = _normalize_values(all_values)

    for xs, ys, zs, cdata in surfaces:
        facecolors = cmap_obj(norm(cdata))
        ls = LightSource(azdeg=315, altdeg=45)
        facecolors = facecolors.copy()
        facecolors[:, :, :3] = ls.shade_rgb(facecolors[:, :, :3], zs)
        ax.plot_surface(
            xs,
            ys,
            zs,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
            **kw,
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=7, pad=-3)
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, fontsize=9, pad=0)
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
    return fig, ax


# ---------------------------------------------------------------------------
# Ballfun surface — @ballfun/surf.m (delegates to plot)
# ---------------------------------------------------------------------------


def surf_ball(
    bf,
    ax=None,
    title: str = "",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Surface plot of a Ballfun (delegates to plot_ball_slices).

    Faithful translation of @ballfun/surf.m which simply calls plot().

    Parameters
    ----------
    bf : Ballfun
    ax : Axes3D, optional
    title : str

    Returns
    -------
    fig, ax
    """
    return plot_ball_slices(bf, ax=ax, title=title, **kw)


# ---------------------------------------------------------------------------
# Ballfunv quiver — @ballfunv/quiver.m
# ---------------------------------------------------------------------------


def quiver_ball(
    bfv,
    ax=None,
    title: str = "",
    n_pts: int = 25,
    arrow_scale: float = 2.5,
    color_by_magnitude: bool = True,
    cmap=None,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Quiver plot of a Ballfunv inside the unit ball (MATLAB @ballfunv/quiver.m).

    Generates arrows at approximately equally-spaced points inside the ball,
    matching the MATLAB algorithm that varies the number of angular samples
    per radial shell.

    Parameters
    ----------
    bfv : Ballfunv
    ax : Axes3D, optional
    title : str
    n_pts : int
        Grid density parameter.
    arrow_scale : float
        Arrow auto-scaling factor.
    color_by_magnitude : bool
        If True, arrows are coloured by magnitude.
    cmap : colormap, optional

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    r = np.asarray(chebpts(n_pts))[n_pts // 2 :]

    xx_list: list[np.ndarray] = []
    yy_list: list[np.ndarray] = []
    zz_list: list[np.ndarray] = []
    vxx_list: list[np.ndarray] = []
    vyy_list: list[np.ndarray] = []
    vzz_list: list[np.ndarray] = []

    fx, fy, fz = bfv.components

    for ri in r:
        nth = max(int(np.ceil(n_pts * ri / 2.0)), 1)
        th_i = np.linspace(0.0, np.pi, nth)
        for thk in th_i:
            dth = min(thk, abs(thk - np.pi))
            nlam = max(int(np.ceil(n_pts * ri * dth * 2.0 / np.pi)), 1)
            lam_i = np.asarray(trigpts(nlam)[0]) * np.pi

            vx = np.asarray(fx.fevalm(jnp.asarray([ri]), jnp.asarray(lam_i), jnp.asarray([thk])))[0, :, 0]
            vy = np.asarray(fy.fevalm(jnp.asarray([ri]), jnp.asarray(lam_i), jnp.asarray([thk])))[0, :, 0]
            vz = np.asarray(fz.fevalm(jnp.asarray([ri]), jnp.asarray(lam_i), jnp.asarray([thk])))[0, :, 0]

            vxx_list.append(np.real(vx))
            vyy_list.append(np.real(vy))
            vzz_list.append(np.real(vz))

            xx_list.append(ri * np.cos(lam_i) * np.sin(thk))
            yy_list.append(ri * np.sin(lam_i) * np.sin(thk))
            zz_list.append(np.full_like(lam_i, ri * np.cos(thk), dtype=float))

    xx = np.concatenate(xx_list)
    yy = np.concatenate(yy_list)
    zz = np.concatenate(zz_list)
    Vxx = np.concatenate(vxx_list)
    Vyy = np.concatenate(vyy_list)
    Vzz = np.concatenate(vzz_list)

    mag = np.sqrt(Vxx ** 2 + Vyy ** 2 + Vzz ** 2)
    max_mag = float(mag.max()) if mag.size and float(mag.max()) > 0 else 1.0
    scale = 0.08 * arrow_scale / max_mag

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5, figsize=(6.1, 2.75))

    q = ax.quiver(
        xx,
        yy,
        zz,
        Vxx * scale,
        Vyy * scale,
        Vzz * scale,
        length=1.0,
        arrow_length_ratio=0.3,
        linewidth=0.6,
    )

    if color_by_magnitude:
        mags = np.sqrt(Vxx ** 2 + Vyy ** 2 + Vzz ** 2)
        max_mag = float(mags.max()) if mags.max() > 0 else 1.0
        norm_mags = mags / max_mag
        # Colour arrows by magnitude
        colors = cmap_obj(norm_mags)
        q.set_color(colors)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Chebfun3 plot — @chebfun3/plot.m (boundary face slices)
# ---------------------------------------------------------------------------


def plot_chebfun3(
    f3,
    ax=None,
    title: str = "",
    n_pts: int = 151,
    cmap=None,
    alpha: float = 0.85,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Plot a Chebfun3 by showing slices on the 6 boundary faces of its box.

    Faithful translation of @chebfun3/plot.m which calls MATLAB's slice()
    at the domain bounds.

    Parameters
    ----------
    f3 : Chebfun3
    ax : Axes3D, optional
    title : str
    n_pts : int
    cmap : colormap, optional
    alpha : float

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    cmap_obj = _coerce_cmap(cmap)

    try:
        xa, xb, ya, yb, za, zb = f3.domain
    except Exception:
        xa, xb, ya, yb, za, zb = -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

    xa, xb = float(xa), float(xb)
    ya, yb = float(ya), float(yb)
    za, zb = float(za), float(zb)

    # Reduced resolution for boundary evaluation
    n = min(n_pts, 80)
    xs = np.linspace(xa, xb, n)
    ys = np.linspace(ya, yb, n)
    zs = np.linspace(za, zb, n)

    slices = []

    # x = xa face (YZ plane)
    YY_yz, ZZ_yz = np.meshgrid(ys, zs, indexing="ij")
    XX_yz = np.full_like(YY_yz, xa)
    F_yz = np.array(
        f3(jnp.array(XX_yz.ravel()), jnp.array(YY_yz.ravel()), jnp.array(ZZ_yz.ravel()))
    ).reshape(YY_yz.shape)
    slices.append((XX_yz, YY_yz, ZZ_yz, F_yz))

    # x = xb face
    XX_yz2 = np.full_like(YY_yz, xb)
    F_yz2 = np.array(
        f3(jnp.array(XX_yz2.ravel()), jnp.array(YY_yz.ravel()), jnp.array(ZZ_yz.ravel()))
    ).reshape(YY_yz.shape)
    slices.append((XX_yz2, YY_yz, ZZ_yz, F_yz2))

    # y = ya face (XZ plane)
    XX_xz, ZZ_xz = np.meshgrid(xs, zs, indexing="ij")
    YY_xz = np.full_like(XX_xz, ya)
    F_xz = np.array(
        f3(jnp.array(XX_xz.ravel()), jnp.array(YY_xz.ravel()), jnp.array(ZZ_xz.ravel()))
    ).reshape(XX_xz.shape)
    slices.append((XX_xz, YY_xz, ZZ_xz, F_xz))

    # y = yb face
    YY_xz2 = np.full_like(XX_xz, yb)
    F_xz2 = np.array(
        f3(jnp.array(XX_xz.ravel()), jnp.array(YY_xz2.ravel()), jnp.array(ZZ_xz.ravel()))
    ).reshape(XX_xz.shape)
    slices.append((XX_xz, YY_xz2, ZZ_xz, F_xz2))

    # z = za face (XY plane)
    XX_xy, YY_xy = np.meshgrid(xs, ys, indexing="ij")
    ZZ_xy = np.full_like(XX_xy, za)
    F_xy = np.array(
        f3(jnp.array(XX_xy.ravel()), jnp.array(YY_xy.ravel()), jnp.array(ZZ_xy.ravel()))
    ).reshape(XX_xy.shape)
    slices.append((XX_xy, YY_xy, ZZ_xy, F_xy))

    # z = zb face
    ZZ_xy2 = np.full_like(XX_xy, zb)
    F_xy2 = np.array(
        f3(jnp.array(XX_xy.ravel()), jnp.array(YY_xy.ravel()), jnp.array(ZZ_xy2.ravel()))
    ).reshape(XX_xy.shape)
    slices.append((XX_xy, YY_xy, ZZ_xy2, F_xy2))

    all_cdata = np.concatenate([np.asarray(s[3]).ravel() for s in slices])
    is_complex = np.iscomplexobj(all_cdata)
    if is_complex:
        mapped = np.angle(-all_cdata)
        cmap_obj = plt.get_cmap("hsv")
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
    else:
        mapped = np.real(all_cdata)
        norm = _normalize_values(mapped)

    fig, ax = _setup_3d_axes(ax, None, elev=30, azim=-127.5, figsize=(6.1, 2.75))

    for XX_s, YY_s, ZZ_s, F_s in slices:
        scalar = np.angle(-F_s) if is_complex else np.real(F_s)
        fc = cmap_obj(norm(scalar))
        ax.plot_surface(
            XX_s,
            YY_s,
            ZZ_s,
            facecolors=fc,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            alpha=alpha,
            shade=False,
        )

    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_zlim(za, zb)
    if not is_complex:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Chebfun3 surf — @chebfun3/surf.m (3 cross-section 2D surfaces)
# ---------------------------------------------------------------------------


def surf_chebfun3(
    f3,
    ax=None,
    title: str = "",
    n_pts: int = 51,
    cmap=None,
    alpha: float = 0.85,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Three orthogonal cross-section surfaces of a Chebfun3.

    Simplified Python translation of @chebfun3/surf.m (which in MATLAB
    provides an interactive GUI with sliders). Here we plot three
    2D cross-sections (one per coordinate pair) at the domain midpoints.

    Parameters
    ----------
    f3 : Chebfun3
    ax : Axes3D, optional
    title : str
    n_pts : int
    cmap : colormap, optional
    alpha : float

    Returns
    -------
    fig, ax

    Notes
    -----
    The MATLAB original uses a GUI with sliders; this function shows
    static cross-sections at the midpoints (matching the initial view).
    """
    # Delegate to the existing plot_slices which does exactly this
    return plot_slices(f3, ax=ax, title=title, n_pts=n_pts, cmap=cmap,
                       alpha=alpha, **kw)


# ---------------------------------------------------------------------------
# Universal plot dispatcher
# ---------------------------------------------------------------------------


def plot_dispatch(obj, *args, **kwargs):
    """Universal plot dispatcher -- works like MATLAB's plot(f).

    Inspects the type of *obj* and calls the appropriate plotting function.

    Parameters
    ----------
    obj : Chebfun, Chebfun2, Spherefun, Spherefunv, Diskfun, or Ballfun
        The object to plot.
    *args, **kwargs
        Forwarded to the appropriate plotting function.

    Returns
    -------
    fig, ax

    Raises
    ------
    TypeError
        If the object type is not recognized.
    """
    from chebfunjax.ballfun.ballfun import Ballfun
    from chebfunjax.ballfun.ballfunv import Ballfunv
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.diskfun.diskfunv import Diskfunv
    from chebfunjax.spherefun.spherefun import Spherefun
    from chebfunjax.spherefun.spherefunv import Spherefunv

    if isinstance(obj, Chebfun):
        return plot_1d(obj, *args, **kwargs)
    elif isinstance(obj, Chebfun2v):
        # Vector field: quiver for 2-component, surf for 3-component
        if len(obj.components) == 3:
            return surf_chebfun2v(obj, *args, **kwargs)
        else:
            return quiver_2d(obj, *args, **kwargs)
    elif isinstance(obj, Chebfun2):
        return surf(obj, *args, **kwargs)
    elif isinstance(obj, Spherefunv):
        return quiver_sphere(obj, *args, **kwargs)
    elif isinstance(obj, Spherefun):
        return plot_sphere(obj, *args, **kwargs)
    elif isinstance(obj, Diskfunv):
        return quiver_disk(obj, *args, **kwargs)
    elif isinstance(obj, Diskfun):
        return plot_disk(obj, *args, **kwargs)
    elif isinstance(obj, Ballfunv):
        return quiver_ball(obj, *args, **kwargs)
    elif isinstance(obj, Ballfun):
        return plot_ball_slices(obj, *args, **kwargs)
    elif isinstance(obj, Chebfun3):
        return plot_chebfun3(obj, *args, **kwargs)
    else:
        raise TypeError(f"Don't know how to plot {type(obj)}")


# ---------------------------------------------------------------------------
# Waterfall / cascade plot for a sequence of Chebfuns
# ---------------------------------------------------------------------------


def waterfall(
    f_list,
    t=None,
    ax=None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "t",
    color: str = CHEBFUN_BLUE,
    n_pts: int = 400,
    alpha: float = 0.85,
    **kw,
):
    """Waterfall (cascade) plot for a sequence of Chebfuns.

    Plots each Chebfun in *f_list* offset in the z-direction for a 3-D
    waterfall effect, visualising time evolution or parameter dependence.

    Parameters
    ----------
    f_list : list of Chebfun
        Sequence of Chebfuns (e.g. time snapshots).
    ax : Axes3D, optional
    title : str
    xlabel, ylabel : str
    color : str
    n_pts : int
    alpha : float

    Returns
    -------
    fig, ax

    Provenance
    ----------
    Inspired by MATLAB Chebfun waterfall. See https://www.chebfun.org/
    """
    import jax.numpy as jnp
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if len(f_list) == 0:
        raise ValueError("waterfall: f_list must be non-empty.")

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if t is not None:
        t_vals = np.asarray(t, dtype=float)
    else:
        t_vals = np.linspace(0.0, 1.0, len(f_list))
    # Accept (and ignore) MATLAB surface-style options that have no
    # line-plot counterpart, e.g. FaceAlpha/FaceColor.
    kw = {k: v for k, v in kw.items()
          if k.lower() not in ("facealpha", "facecolor", "edgecolor")}
    lw = kw.pop("linewidth", kw.pop("LineWidth", 1.4))
    for i, f in enumerate(f_list):
        xs = _domain_points(f, n_pts)
        ys = np.array(f(jnp.array(xs)))
        ax.plot(xs, ys, zs=t_vals[i], zdir="y", color=color, alpha=alpha,
                linewidth=lw, **kw)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_zlabel("f", fontsize=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Roots plot
# ---------------------------------------------------------------------------


def roots_plot(
    f,
    ax=None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "",
    color: str = CHEBFUN_BLUE,
    root_color: str = CHEBFUN_RED,
    root_markersize: float = 8,
    linewidth: float = 1.8,
    n_pts: int = 600,
    **kw,
):
    """Plot a Chebfun with its roots marked as red circles.

    Parameters
    ----------
    f : Chebfun
    ax : optional
    title, xlabel, ylabel : str
    color : str, function line colour
    root_color : str, root marker colour
    root_markersize : float
    linewidth : float
    n_pts : int

    Returns
    -------
    fig, ax

    Provenance
    ----------
    Inspired by MATLAB Chebfun plot+roots workflow. See https://www.chebfun.org/
    """
    import jax.numpy as jnp

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    xs = _domain_points(f, n_pts)
    ys = np.array(f(jnp.array(xs)))
    ax.plot(xs, ys, color=color, linewidth=linewidth)

    roots = np.array(f.roots())
    if roots.shape[0] > 0:
        ax.plot(roots, np.zeros_like(roots), "o",
                color=root_color, markersize=root_markersize,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    _apply_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Spy plot
# ---------------------------------------------------------------------------


def spy(
    A,
    ax=None,
    title: str = "Sparsity pattern",
    markersize: float = 2,
    **kw,
):
    """Visualise the sparsity pattern of a matrix or Linop.

    Parameters
    ----------
    A : array_like or Linop
    ax : optional
    title : str
    markersize : float

    Returns
    -------
    fig, ax

    Provenance
    ----------
    Wraps matplotlib.axes.Axes.spy. See https://www.chebfun.org/
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    try:
        from chebfunjax.operators.linop import Linop
        if isinstance(A, Linop):
            A = np.array(A._assemble(64))
    except (ImportError, Exception):
        pass

    A_np = np.asarray(A)
    ax.spy(A_np, markersize=markersize, **kw)
    if title:
        ax.set_title(title, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Plotregion — Bernstein ellipse
# ---------------------------------------------------------------------------


def plotregion(
    f,
    ax=None,
    title: str = "Region of analyticity",
    color: str = CHEBFUN_BLUE,
    n_pts: int = 300,
    **kw,
):
    """Plot the Bernstein ellipse showing the region of analyticity.

    Parameters
    ----------
    f : Chebfun
    ax : optional
    title : str
    color : str
    n_pts : int

    Returns
    -------
    fig, ax

    Provenance
    ----------
    Inspired by MATLAB Chebfun plotregion. See https://www.chebfun.org/
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    coeffs = np.abs(np.array(f.coeffs))
    n = len(coeffs)

    if n < 4:
        rho = 2.0
    else:
        tail = coeffs[max(n // 2, 1):]
        if np.max(tail) > 1e-16:
            avg_log = np.mean(np.log(np.maximum(tail, 1e-16)))
            rho = max(np.exp(-avg_log / max(len(tail), 1)), 1.01)
        else:
            rho = 2.0
    rho = min(max(rho, 1.01), 100.0)

    theta = np.linspace(0, 2 * np.pi, n_pts)
    z = rho * np.exp(1j * theta)
    w = 0.5 * (z + 1.0 / z)
    x_ell = np.real(w)
    y_ell = np.imag(w)

    a = float(f.domain.a)
    b = float(f.domain.b)
    x_phys = 0.5 * (b - a) * x_ell + 0.5 * (a + b)

    ax.plot(x_phys, y_ell, color=color, linewidth=1.8, **kw)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.fill_between(x_phys, y_ell, alpha=0.08, color=color)
    ax.set_aspect("equal")

    _apply_style(ax, title=title, xlabel="Re(z)", ylabel="Im(z)")
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Arrow plot — parametric curve with direction arrows
# ---------------------------------------------------------------------------


def arrowplot(
    f,
    g=None,
    ax=None,
    title: str = "",
    color=None,
    n_pts: int = 2000,
    multi: int = 1,
    markersize: float = 6.0,
    ystretch: float = 1.0,
    n_arrows: "int | None" = None,
    **kw,
):
    """Chebfun plot with an arrowhead at the end.

    ``arrowplot(f, g)``, for Chebfuns on a common domain, plots the curve
    ``(f, g)`` in the plane with an arrowhead. ``arrowplot(f)`` for a
    complex Chebfun plots ``(real(f), imag(f))``. Passing lists for ``f``
    and ``g`` plots several curves, as a MATLAB quasimatrix does.

    Arrowheads are placed at ``linspace(a, b, multi + 1)[1:]``, so the
    default ``multi=1`` puts a single head at the end of the curve; the
    direction comes from ``f'`` there.

    Parameters
    ----------
    f : Chebfun or list of Chebfun
        x-component, or a complex Chebfun when ``g`` is None.
    g : Chebfun, list of Chebfun, or None
        y-component.
    ax : matplotlib Axes, optional
    title : str
    color : matplotlib color or list, optional
        Defaults to the Chebfun colour cycle.
    n_pts : int
        Points used to draw the curve itself.
    multi : int
        Number of arrowheads (MATLAB ``'multi', n``).
    markersize : float
        Arrowhead size in points (MATLAB ``'markersize'``, default 6).
    ystretch : float
        Multiplies the arrowhead's slope, for rescaled axes (MATLAB
        ``'ystretch'``).
    n_arrows : int, optional
        Deprecated alias for ``multi``.

    Returns
    -------
    fig, ax

    Provenance
    ----------
    MATLAB source : @chebfun/arrowplot.m
    Chebfun commit: 7574c77
    """
    import jax.numpy as jnp

    if n_arrows is not None:
        multi = int(n_arrows)
    if multi < 1:
        raise ValueError(f"arrowplot: multi must be >= 1, got {multi}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    fs = list(f) if isinstance(f, (list, tuple)) else [f]
    if g is None:
        gs = [None] * len(fs)
    elif isinstance(g, (list, tuple)):
        gs = list(g)
    else:
        gs = [g]
    if len(gs) != len(fs):
        raise ValueError(
            f"arrowplot: got {len(fs)} x-components but {len(gs)} "
            f"y-components.")

    if color is None:
        cyc = [CHEBFUN_BLUE, CHEBFUN_RED, CHEBFUN_GREEN, CHEBFUN_ORANGE,
               "#8B008B", "#008080"]
        colors = [cyc[k % len(cyc)] for k in range(len(fs))]
    elif isinstance(color, (list, tuple)) and not isinstance(color, str):
        colors = list(color)
    else:
        colors = [color] * len(fs)

    for fk, gk, ck in zip(fs, gs, colors):
        ts = _domain_points(fk, n_pts)
        fvals = np.asarray(fk(jnp.array(ts)))
        if gk is not None:
            xvals, yvals = np.real(fvals), np.asarray(gk(jnp.array(ts)))
        else:
            xvals, yvals = np.real(fvals), np.imag(fvals)

        ax.plot(xvals, yvals, color=ck, **kw)

        # MATLAB evaluates f and f' at linspace(a, b, multi+1) minus the
        # first point, so multi=1 gives one arrow at the right endpoint.
        pts = np.linspace(float(ts[0]), float(ts[-1]), multi + 1)[1:]
        fp = fk.diff()
        gp = gk.diff() if gk is not None else None
        for p in pts:
            xp = float(np.real(np.asarray(fp(jnp.float64(p)))))
            if gp is not None:
                yp = float(np.asarray(gp(jnp.float64(p))))
            else:
                yp = float(np.imag(np.asarray(fp(jnp.float64(p)))))
            nrm = np.hypot(xp, yp)
            if nrm == 0.0:
                continue                    # zero chebfun: no arrowhead
            xp, yp = 0.001 * xp / nrm, 0.001 * yp / nrm
            x0 = float(np.real(np.asarray(fk(jnp.float64(p)))))
            y0 = (float(np.asarray(gk(jnp.float64(p)))) if gk is not None
                  else float(np.imag(np.asarray(fk(jnp.float64(p))))))
            ax.annotate(
                "",
                xy=(x0 + xp, y0 + ystretch * yp),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=f"-|>,head_length={markersize / 2},"
                               f"head_width={markersize / 4}",
                    mutation_scale=1, color=ck, lw=kw.get("linewidth", 1.0),
                    shrinkA=0, shrinkB=0),
            )

    if title:
        ax.set_title(title)
    fig.set_facecolor("white")
    return fig, ax


# ---------------------------------------------------------------------------
# chebpolyplot — enhanced coefficient plot with envelope
# ---------------------------------------------------------------------------


def chebpolyplot(
    f,
    ax=None,
    title: str = "Chebyshev polynomial coefficients",
    color: str = CHEBFUN_BLUE,
    envelope_color: str = CHEBFUN_ORANGE,
    **kw,
):
    """Log-scale Chebyshev coefficient plot with envelope line.

    Parameters
    ----------
    f : Chebfun
    ax : optional
    title : str
    color : str  (dots colour)
    envelope_color : str  (envelope line colour)

    Returns
    -------
    fig, ax

    Provenance
    ----------
    Inspired by MATLAB Chebfun plotcoeffs. See https://www.chebfun.org/
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    coeffs = np.abs(np.array(f.coeffs))
    ns = np.arange(len(coeffs))

    ax.semilogy(ns, coeffs, ".", color=color, markersize=5, **kw)

    if len(coeffs) > 2:
        running_max = np.maximum.accumulate(coeffs[::-1])[::-1]
        ax.semilogy(ns, running_max, "-", color=envelope_color,
                    alpha=0.7, linewidth=1.4, label="envelope")

    eps_floor = np.finfo(np.float64).eps * (coeffs[0] if coeffs[0] > 0 else 1.0)
    ax.axhline(eps_floor, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.6, label=r"$\epsilon_{\rm mach} \|f\|$")

    _apply_style(ax, title=title, xlabel="degree $n$", ylabel="$|a_n|$")
    ax.set_ylim(bottom=max(coeffs.min() * 0.1, 1e-18))
    ax.legend(fontsize=8)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# MATLAB-style plot dispatchers (@chebfun/plot.m, plot3.m, surf.m, mesh.m)
# ---------------------------------------------------------------------------

def _cheb_cols(obj):
    """Columns of a Chebfun-like object: Chebfun -> [f]; Quasimatrix ->
    list of its column chebfuns; anything else -> None."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    if isinstance(obj, Chebfun):
        return [obj]
    cols = getattr(obj, "columns", None)
    if cols is None:
        cols = getattr(obj, "cols", None)
        if cols is not None and not all(
                hasattr(c, "funs") for c in cols):
            cols = None
    if cols is not None:
        return list(cols)
    if isinstance(obj, (list, tuple)) and obj and all(
            isinstance(c, Chebfun) for c in obj):
        return list(obj)
    return None


def _sample_pieces(f, numpts: int = 2001, interval=None):
    """Sample a (possibly piecewise / unbounded / singular) Chebfun.

    Returns a list of (x, y) numpy arrays, one per smooth piece, with
    infinite endpoints clipped to a finite window and singular endpoints
    approached from the interior (MATLAB plots blow-ups the same way).
    """
    out = []
    for p in f.funs:
        a, b = float(p.interval[0]), float(p.interval[1])
        if interval is not None:
            lo, hi = float(interval[0]), float(interval[1])
            a, b = max(a, lo), min(b, hi)
            if a >= b:
                continue
        if not np.isfinite(a):
            a = min(-10.0, b - 10.0) if np.isfinite(b) else -10.0
        if not np.isfinite(b):
            b = max(10.0, a + 10.0)
        pad = 1e-8 * max(1.0, abs(b - a))
        n = max(16, numpts // max(1, len(f.funs)))
        x = np.linspace(a + pad, b - pad, n)
        with np.errstate(all="ignore"):
            y = np.asarray(f(jnp.asarray(x)))
        out.append((x, y))
    return out


def _plot_curve(ax, xs, ys, fmt, kw):
    """ax.plot with an optional MATLAB linespec (matplotlib-compatible)."""
    if fmt:
        ax.plot(xs, ys, fmt, **kw)
    else:
        ax.plot(xs, ys, **kw)


def _sing_ylim(ys, exps):
    """MATLAB @singfun/plotData getYLimits: y-limits from the standard
    deviation of the values away from the singular endpoint(s).

    Provenance
    ----------
    MATLAB source : @singfun/plotData.m  (getYLimits subfunction)
    Chebfun commit: 7574c77
    """
    v = ys[np.isfinite(ys)]
    n = v.size
    if n == 0:
        return None
    mask = np.ones(n, dtype=bool)
    if exps[0] < 0:
        scl = min(-0.2 * exps[0], 0.5)
        mask[: max(int(np.ceil(scl * n)), 5)] = False
    if exps[1] < 0:
        scl = min(-0.2 * exps[1], 0.5)
        k = max(int(np.ceil(scl * n)), 5)
        mask[n - k:] = False
    m = v[mask]
    if m.size == 0:
        return None
    sd = float(np.std(m, ddof=1)) if m.size > 1 else 0.0
    bot = max(float(v.min()), float(m.min()) - sd)
    top = min(float(v.max()), float(m.max()) + sd)
    return bot, top


def _function_lims(f, numpts=2001, interval=None):
    """(xLim, yLim, default_ylim) for one chebfun, per MATLAB plotData:
    the x-limits are the (windowed) domain -- unbounded endpoints are
    clipped to a width-10 window; smooth pieces contribute their actual
    value range to yLim with the default flag kept, while singular
    (exps < 0) pieces contribute the std-based getYLimits suggestion
    and unset defaultYLim.

    Provenance
    ----------
    MATLAB source : @chebfun/plotData.m, @bndfun/plotData.m,
        @unbndfun/plotData.m, @singfun/plotData.m
    Chebfun commit: 7574c77
    """
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]
    default_ylim = True
    a0 = float(f.domain.a)
    b0 = float(f.domain.b)
    window = 10.0
    if not np.isfinite(a0) and not np.isfinite(b0):
        wlo, whi = -window, window
    elif not np.isfinite(a0):
        wlo, whi = b0 - window, b0
    elif not np.isfinite(b0):
        wlo, whi = a0, a0 + window
    else:
        wlo, whi = a0, b0
    for p in f.funs:
        a, b = float(p.interval[0]), float(p.interval[1])
        a = wlo if not np.isfinite(a) else max(a, wlo)
        b = whi if not np.isfinite(b) else min(b, whi)
        if b <= a:
            continue
        exps = getattr(p.tech, "exponents", None)
        pad = 1e-8 * max(1.0, abs(b - a))
        x = np.linspace(a + pad, b - pad, 1001)
        with np.errstate(all="ignore"):
            y = np.asarray(f(jnp.asarray(x)), dtype=float)
        if exps is not None and (float(exps[0]) < 0 or float(exps[1]) < 0):
            yl = _sing_ylim(y, (float(exps[0]), float(exps[1])))
            if yl is not None:
                default_ylim = False
                ylim[0] = min(ylim[0], yl[0])
                ylim[1] = max(ylim[1], yl[1])
        else:
            yf = y[np.isfinite(y)]
            if yf.size:
                ylim[0] = min(ylim[0], float(yf.min()))
                ylim[1] = max(ylim[1], float(yf.max()))
        xlim[0] = min(xlim[0], a)
        xlim[1] = max(xlim[1], b)
    if interval is not None:
        xlim = [float(interval[0]), float(interval[1])]
    return xlim, ylim, default_ylim


def _apply_matlab_lims(ax, xlim, ylim, default_ylim, entry_lims):
    """Apply MATLAB @chebfun/plot.m axis-limit policy:

    - x-limits are always set (unioned with the ENTRY limits when
      holding -- ``entry_lims`` is (xlim, ylim, ymanual) captured
      before this call plotted anything, or None when not holding);
    - y-limits are set only when a blow-up suggested them, or when
      holding onto axes whose y-limits were already set manually --
      otherwise matplotlib's auto mode is left in charge.

    Provenance
    ----------
    MATLAB source : @chebfun/plot.m
    Chebfun commit: 7574c77
    """
    hold = entry_lims is not None
    ymanual_prev = entry_lims[2] if hold else False
    if hold:
        cx, cy, _ = entry_lims
        xlim = [min(cx[0], xlim[0]), max(cx[1], xlim[1])]
        ylim = [min(cy[0], ylim[0]), max(cy[1], ylim[1])]
    if np.isfinite(xlim[0]) and np.isfinite(xlim[1]) and xlim[1] > xlim[0]:
        ax.set_xlim(xlim)
    if (not default_ylim or (hold and ymanual_prev)) \
            and np.isfinite(ylim[0]) and np.isfinite(ylim[1]) \
            and ylim[1] > ylim[0]:
        ax.set_ylim(ylim)


def _draw_jumplines(ax, f, jumpline, kw):
    if jumpline is None or jumpline == "none":
        return
    style_kw = dict(kw)
    fmt = None
    if isinstance(jumpline, str):
        fmt = jumpline
    elif isinstance(jumpline, dict):
        style_kw.update({k.lower(): v for k, v in jumpline.items()})
    elif isinstance(jumpline, (list, tuple)):
        it = iter(jumpline)
        for k in it:
            style_kw[str(k).lower()] = next(it)
    style_kw = {("linestyle" if k in ("linestyle", "lines") else k): v
                for k, v in style_kw.items()}
    breaks = [float(p.interval[1]) for p in f.funs[:-1]]
    eps_ = 1e-10
    for xb in breaks:
        try:
            yl = float(f(jnp.asarray(xb - eps_)))
            yr = float(f(jnp.asarray(xb + eps_)))
        except Exception:
            continue
        if fmt:
            ax.plot([xb, xb], [yl, yr], fmt, **style_kw)
        else:
            style_kw.setdefault("linestyle", ":")
            ax.plot([xb, xb], [yl, yr], **style_kw)


def _draw_deltas(ax, f, deltaline, kw):
    deltas = getattr(f, "deltas", ()) or ()
    if not deltas:
        return
    fmt = deltaline if isinstance(deltaline, str) else None
    for d in deltas:
        loc, mag = float(d[0]), float(d[1])
        if fmt:
            ax.plot([loc, loc], [0.0, mag], fmt, **kw)
        else:
            ax.plot([loc, loc], [0.0, mag], "-", **kw)
        marker = "^" if mag >= 0 else "v"
        ax.plot([loc], [mag], marker, **kw)


def matlab_plot(*args, ax=None, numpts: int = 2001, interval=None,
                jumpline=None, deltaline=None, **kw):
    """MATLAB @chebfun/plot.m argument-stream plotting.

    Supports the argument forms exercised by tests/chebfun/test_plot.m:
    ``plot(f)``, ``plot(f, style)``, ``plot(f, g[, style])`` (parametric),
    ``plot(f, [a b])`` (interval shorthand), mixes of quasimatrices and
    scalar chebfuns (columns broadcast), discrete data groups
    ``plot(x, y, style)``, complex chebfuns (real vs imag), and the
    ``numpts`` / ``interval`` / ``jumpline`` / ``deltaline`` options
    (passed as Python keywords).

    Provenance
    ----------
    MATLAB source : @chebfun/plot.m
    Chebfun commit: 7574c77
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        entry_lims = None
    else:
        fig = ax.get_figure()
        # MATLAB stores the current limits (and the ylim mode) on ENTRY,
        # before any new data lands on the axes -- the hold-union must
        # not see the raw values of curves plotted by this very call
        # (a blow-up's ~1e6 samples would swamp the union).
        entry_lims = (tuple(ax.get_xlim()), tuple(ax.get_ylim()),
                      not ax.get_autoscaley_on())

    agg_x = [np.inf, -np.inf]
    agg_y = [np.inf, -np.inf]
    agg_default = True
    saw_plain = False

    items = list(args)
    i = 0
    while i < len(items):
        obj = items[i]
        cols = _cheb_cols(obj)
        if cols is not None:
            ycols = None
            fmt = None
            j = i + 1
            if j < len(items):
                nxt = _cheb_cols(items[j])
                if nxt is not None:
                    ycols = nxt
                    j += 1
                elif isinstance(items[j], (list, tuple, np.ndarray)) \
                        and np.asarray(items[j]).size == 2 \
                        and not isinstance(items[j], str):
                    interval = np.asarray(items[j], dtype=float)
                    j += 1
            if j < len(items) and isinstance(items[j], str):
                fmt = items[j]
                j += 1
            if ycols is None:
                for f in cols:
                    was_complex = False
                    for xs, ys in _sample_pieces(f, numpts, interval):
                        if np.iscomplexobj(ys):
                            was_complex = True
                            _plot_curve(ax, ys.real, ys.imag, fmt, kw)
                        else:
                            _plot_curve(ax, xs, ys, fmt, kw)
                    _draw_jumplines(ax, f, jumpline, kw)
                    _draw_deltas(ax, f, deltaline, kw)
                    if not was_complex:
                        saw_plain = True
                        fx, fy, fd = _function_lims(f, numpts, interval)
                        agg_x = [min(agg_x[0], fx[0]), max(agg_x[1], fx[1])]
                        agg_y = [min(agg_y[0], fy[0]),
                                 max(agg_y[1], fy[1])]
                        agg_default = agg_default and fd
            else:
                nx, ny = len(cols), len(ycols)
                npairs = max(nx, ny)
                for k in range(npairs):
                    fx = cols[k % nx]
                    fy = ycols[k % ny]
                    for (xs, xv), (_, yv) in zip(
                            _sample_pieces(fx, numpts, interval),
                            _sample_pieces(fy, numpts, interval)):
                        _plot_curve(ax, xv, yv, fmt, kw)
            i = j
        else:
            # Discrete data group: x, y[, style]
            xd = np.asarray(obj, dtype=float)
            yd = np.asarray(items[i + 1])
            fmt = None
            j = i + 2
            if j < len(items) and isinstance(items[j], str):
                fmt = items[j]
                j += 1
            _plot_curve(ax, xd, yd, fmt, kw)
            i = j
    if saw_plain:
        _apply_matlab_lims(ax, agg_x, agg_y, agg_default, entry_lims)
    return fig, ax


def matlab_plot3(*args, ax=None, numpts: int = 2001, jumpline=None, **kw):
    """MATLAB @chebfun/plot3.m: 3-D parametric curves of chebfun triples,
    with quasimatrix columns broadcast.

    Provenance
    ----------
    MATLAB source : @chebfun/plot3.m
    Chebfun commit: 7574c77
    """
    triple = []
    fmt = None
    for obj in args:
        cols = _cheb_cols(obj)
        if cols is not None and len(triple) < 3:
            triple.append(cols)
        elif isinstance(obj, str) and fmt is None:
            fmt = obj
    if len(triple) != 3:
        raise ValueError("plot3 requires three chebfun arguments")
    if ax is None:
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    ncols = max(len(c) for c in triple)
    for k in range(ncols):
        fx = triple[0][k % len(triple[0])]
        fy = triple[1][k % len(triple[1])]
        fz = triple[2][k % len(triple[2])]
        for (t, xv), (_, yv), (_, zv) in zip(
                _sample_pieces(fx, numpts), _sample_pieces(fy, numpts),
                _sample_pieces(fz, numpts)):
            if fmt:
                ax.plot(xv, yv, zv, fmt, **kw)
            else:
                ax.plot(xv, yv, zv, **kw)
    return fig, ax


def matlab_surf_quasi(F, ax=None, mode: str = "surf", numpts: int = 201,
                      **kw):
    """MATLAB @chebfun/surf.m family for array-valued chebfuns /
    quasimatrices: surface z = F_j(x) over (x, column index j).
    ``mode`` selects 'surf', 'surfc', or 'mesh' ('surface' is a wrapper
    for 'surf', as in MATLAB).

    Provenance
    ----------
    MATLAB source : @chebfun/surf.m, @chebfun/mesh.m, @chebfun/surfc.m
    Chebfun commit: 7574c77
    """
    cols = _cheb_cols(F)
    if cols is None:
        raise TypeError("surf expects an array-valued chebfun/quasimatrix")
    dom = cols[0].domain
    a, b = float(dom.a), float(dom.b)
    x = np.linspace(a + 1e-9, b - 1e-9, numpts)
    Z = np.vstack([np.asarray(c(jnp.asarray(x))) for c in cols])
    X, Y = np.meshgrid(x, np.arange(1, len(cols) + 1))
    if ax is None:
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    if mode == "mesh":
        ax.plot_wireframe(X, Y, np.real(Z), **kw)
    else:
        ax.plot_surface(X, Y, np.real(Z), **kw)
        if mode == "surfc":
            try:
                ax.contour(X, Y, np.real(Z),
                           offset=float(np.real(Z).min()) - 0.5)
            except Exception:
                pass
    return fig, ax


def comet(f, g=None, ax=None, numpts: int = 501, **kw):
    """MATLAB @chebfun/comet.m: animated trace of a curve.  Headless
    (Agg) rendering draws the full trace in one shot; with an
    interactive backend the curve is drawn progressively.

    ``comet(f)`` traces y = f(x); ``comet(f, g)`` traces the parametric
    curve (f(t), g(t)).

    Provenance
    ----------
    MATLAB source : @chebfun/comet.m
    Chebfun commit: 7574c77
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()
    fs = _cheb_cols(f)[0]
    if g is None:
        pts = [(xs, ys) for xs, ys in _sample_pieces(fs, numpts)]
    else:
        gs = _cheb_cols(g)[0]
        pts = [(xv, yv) for (_, xv), (_, yv) in zip(
            _sample_pieces(fs, numpts), _sample_pieces(gs, numpts))]
    for xs, ys in pts:
        ax.plot(xs, ys, **kw)
    return fig, ax


def comet3(f, g=None, h=None, ax=None, numpts: int = 501, **kw):
    """MATLAB @chebfun/comet3.m: animated 3-D trace.  ``comet3(f, g, h)``
    traces (f, g, h); ``comet3(F)`` with a three-column array-valued
    chebfun / quasimatrix traces its columns.

    Provenance
    ----------
    MATLAB source : @chebfun/comet3.m
    Chebfun commit: 7574c77
    """
    if g is None:
        cols = _cheb_cols(f)
        if cols is None or len(cols) < 3:
            raise ValueError("comet3 requires three curves")
        f, g, h = cols[0], cols[1], cols[2]
    if ax is None:
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    for (t, xv), (_, yv), (_, zv) in zip(
            _sample_pieces(_cheb_cols(f)[0], numpts),
            _sample_pieces(_cheb_cols(g)[0], numpts),
            _sample_pieces(_cheb_cols(h)[0], numpts)):
        ax.plot(xv, yv, zv, **kw)
    return fig, ax


def contour3_disk(fd, ax=None, levels: int = 10, n_pts: int = 200,
                  pivots=None, xx=None, yy=None, **kw):
    """3-D contour plot of a Diskfun: contour curves drawn at their
    function height above the unit disk (MATLAB @diskfun/contour3.m).

    ``pivots`` overlays the construction pivot locations with the given
    linespec; ``xx``/``yy`` give an explicit Cartesian evaluation grid.

    Provenance
    ----------
    MATLAB source : @diskfun/contour3.m
    Chebfun commit: 7574c77
    """
    if not np.isscalar(levels):
        lv = np.atleast_1d(np.asarray(levels, dtype=float))
        if lv.size == 2 and lv[0] == lv[1]:
            levels = [float(lv[0])]
    if xx is not None and yy is not None:
        XX = np.asarray(xx, dtype=float)
        YY = np.asarray(yy, dtype=float)
    else:
        g = np.linspace(-1.0, 1.0, n_pts)
        XX, YY = np.meshgrid(g, g)
    RR = np.hypot(XX, YY)
    TT = np.arctan2(YY, XX)
    Rc = np.clip(RR, 0.0, 1.0)
    ZZ = np.asarray(fd(jnp.asarray(TT), jnp.asarray(Rc)), dtype=float)
    ZZ = np.where(RR <= 1.0, ZZ, np.nan)

    # Extract 2-D contour paths, then draw each at its level height.
    ftmp, axtmp = plt.subplots()
    cs = axtmp.contour(XX, YY, ZZ, levels=levels)
    paths = []
    for i, segs in enumerate(cs.allsegs):
        lev = cs.levels[i] if i < len(cs.levels) else cs.levels[-1]
        for seg in segs:
            if len(seg) > 1:
                paths.append((lev, seg))
    plt.close(ftmp)

    if ax is None:
        fig = plt.figure(figsize=(6.1, 4.0))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    cmap_obj = _coerce_cmap(None)
    lvs = cs.levels
    span = (lvs[-1] - lvs[0]) if len(lvs) > 1 and lvs[-1] > lvs[0] else 1.0
    for lev, seg in paths:
        c = cmap_obj((lev - lvs[0]) / span)
        ax.plot(seg[:, 0], seg[:, 1], zs=lev, zdir="z", color=c, **kw)
    tb = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tb), np.sin(tb), zs=0.0, zdir="z",
            color="k", linewidth=0.6, alpha=0.5)
    if pivots is not None:
        locs = np.asarray([(float(p[0]), float(p[1]))
                           for p in getattr(fd, "pivot_locations", ())],
                          dtype=float)
        if locs.size:
            fmt_p = pivots if isinstance(pivots, str) else "r."
            px = locs[:, 1] * np.cos(locs[:, 0])
            py = locs[:, 1] * np.sin(locs[:, 0])
            pz = np.asarray(fd(jnp.asarray(locs[:, 0]),
                               jnp.asarray(locs[:, 1])), dtype=float)
            ax.plot(px, py, pz, fmt_p)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    return fig, ax


def slice_chebfun3(f3, xslices=None, yslices=None, zslices=None,
                   ax=None, n_pts: int = 101, cmap=None, alpha: float = 0.9,
                   **kw):
    """Orthogonal slice-plane plot of a Chebfun3 (MATLAB @chebfun3/slice.m,
    'noslider' variant -- there is no interactive slider headlessly).

    Default slice positions are the midpoints of the domain box.

    Provenance
    ----------
    MATLAB source : @chebfun3/slice.m
    Chebfun commit: 7574c77
    """
    cmap_obj = _coerce_cmap(cmap)
    xa, xb, ya, yb, za, zb = (float(v) for v in f3.domain)
    if xslices is None:
        xslices = [0.5 * (xa + xb)]
    if yslices is None:
        yslices = [0.5 * (ya + yb)]
    if zslices is None:
        zslices = [0.5 * (za + zb)]
    xslices = np.atleast_1d(np.asarray(xslices, dtype=float))
    yslices = np.atleast_1d(np.asarray(yslices, dtype=float))
    zslices = np.atleast_1d(np.asarray(zslices, dtype=float))

    if ax is None:
        fig = plt.figure(figsize=(6.1, 4.6))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()

    # Common normalization across all slices.
    vals_all = []
    grids = []
    u = np.linspace(0.0, 1.0, n_pts)
    for xs in xslices:
        Y, Z = np.meshgrid(ya + (yb - ya) * u, za + (zb - za) * u)
        X = np.full_like(Y, xs)
        V = np.asarray(f3(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)),
                       dtype=float)
        vals_all.append(V)
        grids.append((X, Y, Z, V))
    for ys in yslices:
        X, Z = np.meshgrid(xa + (xb - xa) * u, za + (zb - za) * u)
        Y = np.full_like(X, ys)
        V = np.asarray(f3(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)),
                       dtype=float)
        vals_all.append(V)
        grids.append((X, Y, Z, V))
    for zs in zslices:
        X, Y = np.meshgrid(xa + (xb - xa) * u, ya + (yb - ya) * u)
        Z = np.full_like(X, zs)
        V = np.asarray(f3(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)),
                       dtype=float)
        vals_all.append(V)
        grids.append((X, Y, Z, V))
    norm = _normalize_values(np.concatenate([v.ravel() for v in vals_all]))
    for X, Y, Z, V in grids:
        ax.plot_surface(X, Y, Z, facecolors=cmap_obj(norm(V)),
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False, alpha=alpha, **kw)
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_zlim(za, zb)
    return fig, ax


def scan_chebfun3(f3, dim: int = 1, hold: bool = False, ax=None,
                  n_frames: int = 5, n_pts: int = 81, **kw):
    """Scan plot of a Chebfun3: a sequence of slices moving through the
    domain along dimension ``dim`` (MATLAB @chebfun3/scan.m; the
    animation renders as superimposed frames headlessly).

    Provenance
    ----------
    MATLAB source : @chebfun3/scan.m
    Chebfun commit: 7574c77
    """
    xa, xb, ya, yb, za, zb = (float(v) for v in f3.domain)
    lo, hi = ((xa, xb), (ya, yb), (za, zb))[dim - 1]
    frames = np.linspace(lo, hi, n_frames + 2)[1:-1]
    fig = None
    for i, s in enumerate(frames):
        kwargs = {("xslices", "yslices", "zslices")[dim - 1]: [float(s)]}
        if hold and fig is not None:
            slice_chebfun3(f3, ax=ax, n_pts=n_pts, **kwargs, **kw)
        else:
            fig, ax = slice_chebfun3(f3, n_pts=n_pts, **kwargs, **kw)
        if not hold and i < len(frames) - 1:
            plt.close(fig)
    return fig, ax


def isosurface_chebfun3(f3, levels=None, ax=None, n_pts: int = 51,
                        cmap=None, alpha: float = 0.8, **kw):
    """Isosurface plot of a Chebfun3 via marching cubes (MATLAB
    @chebfun3/isosurface.m, 'noslider' variant).

    ``levels`` may be a scalar or a sequence; the default is the
    midpoint between the sampled min and max.

    Provenance
    ----------
    MATLAB source : @chebfun3/isosurface.m
    Chebfun commit: 7574c77
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    cmap_obj = _coerce_cmap(cmap)
    xa, xb, ya, yb, za, zb = (float(v) for v in f3.domain)
    u = np.linspace(0.0, 1.0, n_pts)
    X, Y, Z = np.meshgrid(xa + (xb - xa) * u, ya + (yb - ya) * u,
                          za + (zb - za) * u, indexing="ij")
    V = np.asarray(f3(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)),
                   dtype=float)
    vmin, vmax = float(V.min()), float(V.max())
    if levels is None:
        levels = [0.5 * (vmin + vmax)]
    levels = np.atleast_1d(np.asarray(levels, dtype=float))

    if ax is None:
        fig = plt.figure(figsize=(6.1, 4.6))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    span = max(vmax - vmin, np.finfo(float).tiny)
    spacing = ((xb - xa) / (n_pts - 1), (yb - ya) / (n_pts - 1),
               (zb - za) / (n_pts - 1))
    for lev in levels:
        lv = min(max(float(lev), vmin + 1e-12 * span), vmax - 1e-12 * span)
        try:
            verts, faces, _, _ = measure.marching_cubes(
                V, level=lv, spacing=spacing)
        except (ValueError, RuntimeError):
            continue
        verts = verts + np.array([xa, ya, za])
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(cmap_obj((lv - vmin) / span))
        ax.add_collection3d(mesh)
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_zlim(za, zb)
    return fig, ax


def waterfall_chebfun2(f2, fmt=None, ax=None, n_lines: int = 20,
                       n_pts: int = 121, **kw):
    """Waterfall plot of a Chebfun2: 3-D line traces of the surface
    along constant-y sections (MATLAB @separableApprox/waterfall.m).

    Provenance
    ----------
    MATLAB source : @separableApprox/waterfall.m
    Chebfun commit: 7574c77
    """
    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0
    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_lines)
    if ax is None:
        fig = plt.figure(figsize=(6.1, 4.0))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    for yv in ys:
        XX, YY = np.meshgrid(xs, [yv])
        Z = _eval_2d_vectorized(f2, XX, YY).ravel()
        if fmt:
            ax.plot(xs, np.full_like(xs, yv), Z, fmt, **kw)
        else:
            ax.plot(xs, np.full_like(xs, yv), Z, color=CHEBFUN_BLUE, **kw)
    ax.set_xlim(float(x0), float(x1))
    ax.set_ylim(float(y0), float(y1))
    return fig, ax
