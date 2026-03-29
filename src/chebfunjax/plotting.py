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
from typing import Any, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib
# Do NOT call matplotlib.use("Agg") unconditionally — that breaks Jupyter
# inline plotting.  Only switch if we are already headless or truly have no
# display available.
if matplotlib.get_backend().lower() == "agg" and not os.environ.get("DISPLAY"):
    pass  # already headless — keep whatever backend is active
import matplotlib.pyplot as plt
import numpy as np
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


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CHEBFUN_BLUE = "#0072BD"   # MATLAB default blue
CHEBFUN_RED  = "#D95319"   # MATLAB default orange/red
CHEBFUN_GREEN = "#77AC30"  # MATLAB default green
CHEBFUN_ORANGE = "#EDB120" # MATLAB default yellow/orange

_DEFAULT_LINE_KW: dict[str, Any] = dict(color=CHEBFUN_BLUE, linewidth=1.2)
_DEFAULT_GRID_KW: dict[str, Any] = dict(alpha=0.3, linestyle="--", linewidth=0.6)


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


def _domain_points(f, n: int = 600) -> np.ndarray:
    """Return *n* equispaced points spanning the domain of a Chebfun."""
    # Works for both Chebfun1D (f.domain.breakpoints) and plain Chebfun2.
    try:
        bp = f.domain.breakpoints
        return np.linspace(float(bp[0]), float(bp[-1]), n)
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


def _setup_3d_axes(ax, fig, elev=25, azim=-37, figsize=(6.1, 2.58)):
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

    Returns
    -------
    fig, ax
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=figsize)
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
        xs = _domain_points(f, n_pts)
        ys = np.array(f(jnp.array(xs)))

        c = color if idx == 0 else _colors[idx % len(_colors)]
        plot_kw: dict[str, Any] = dict(color=c, linewidth=linewidth,
                                       linestyle=linestyle)
        if label and idx == 0:
            plot_kw["label"] = label
        # Forward extra kwargs only to first series to avoid clashes
        if idx == 0:
            plot_kw.update(kw)
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

    coeffs = np.abs(np.array(f.coeffs))
    ns = np.arange(len(coeffs))

    ax.semilogy(ns, coeffs, ".", color=color, markersize=4, **kw)

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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37,
                             figsize=(6.1, 2.58))

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
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Contour plot of a Chebfun2 (MATLAB Chebfun style).

    By default draws filled contours with black contour lines overlaid,
    using the parula colormap and unit-domain ticks.

    Parameters
    ----------
    f2 : Chebfun2
    ax : Axes, optional
    title : str
    n_pts : int
    levels : int
    cmap : colormap, optional  (default: parula)
    filled : bool
        If True (default), use contourf + contour overlay.  If False,
        contour lines only.

    Returns
    -------
    fig, ax
    """
    cmap_obj = _coerce_cmap(cmap)

    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = _eval_2d_vectorized(f2, XX, YY)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.1, 2.58))
    else:
        fig = ax.get_figure()

    if filled:
        ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap_obj, **kw)
    ax.contour(XX, YY, ZZ, levels=levels, cmap=cmap_obj, linewidths=0.8, **kw)

    ax.set_aspect("equal")
    _set_unit_ticks(ax, domain=(x0, x1, y0, y1))
    _apply_style(ax, title=title, grid=False)
    fig.set_facecolor("white")
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
        fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37,
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
                                 figsize=(6.1, 2.75))

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
        # Color from level
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
    import jax.numpy as jnp

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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37, figsize=(6.1, 2.75))

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

    ax.contour(XX, YY, vals, levels=levels, cmap=cmap_obj, **kw)

    _draw_disk_boundary(ax, linewidth=0.3)

    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37, figsize=(6.1, 2.75))

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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37,
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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37,
                             figsize=(6.1, 2.58))

    try:
        from skimage.measure import marching_cubes
        _have_skimage = True
    except ImportError:
        _have_skimage = False

    if _have_skimage:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.colors as mcolors

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

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37, figsize=(6.1, 2.75))

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

    fig, ax = _setup_3d_axes(ax, None, elev=25, azim=-37, figsize=(6.1, 2.75))

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
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    from chebfunjax.spherefun.spherefun import Spherefun
    from chebfunjax.spherefun.spherefunv import Spherefunv
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.diskfun.diskfunv import Diskfunv
    from chebfunjax.ballfun.ballfun import Ballfun
    from chebfunjax.ballfun.ballfunv import Ballfunv
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

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

    t_vals = np.linspace(0.0, 1.0, len(f_list))
    for i, f in enumerate(f_list):
        xs = _domain_points(f, n_pts)
        ys = np.array(f(jnp.array(xs)))
        ax.plot(xs, ys, zs=t_vals[i], zdir="y", color=color, alpha=alpha,
                linewidth=1.4, **kw)

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
    color: str = CHEBFUN_BLUE,
    n_pts: int = 400,
    n_arrows: int = 12,
    **kw,
):
    """Parametric curve plot with direction arrows.

    Parameters
    ----------
    f : Chebfun  (x-component or complex Chebfun)
    g : Chebfun or None  (y-component; if None treat f as complex)
    ax : optional
    title : str
    color : str
    n_pts : int
    n_arrows : int

    Returns
    -------
    fig, ax

    Provenance
    ----------
    Inspired by MATLAB Chebfun arrowplot. See https://www.chebfun.org/
    """
    import jax.numpy as jnp

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    ts = _domain_points(f, n_pts)
    fvals = np.array(f(jnp.array(ts)))

    if g is not None:
        xvals = fvals
        yvals = np.array(g(jnp.array(ts)))
    else:
        xvals = np.real(fvals)
        yvals = np.imag(fvals)

    ax.plot(xvals, yvals, color=color, linewidth=1.8, **kw)

    arrow_idx = np.linspace(n_pts // (n_arrows + 1),
                            n_pts - n_pts // (n_arrows + 1),
                            n_arrows, dtype=int)
    for i in arrow_idx:
        if i + 1 >= n_pts:
            continue
        dx = xvals[i + 1] - xvals[i]
        dy = yvals[i + 1] - yvals[i]
        ax.annotate(
            "",
            xy=(xvals[i] + dx * 0.01, yvals[i] + dy * 0.01),
            xytext=(xvals[i], yvals[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.set_aspect("equal")
    _apply_style(ax, title=title, xlabel="Re", ylabel="Im")
    fig.set_facecolor("white")
    fig.tight_layout()
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
