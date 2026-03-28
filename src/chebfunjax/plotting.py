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
    filled: bool = True,
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
    if cmap is None:
        cmap = PARULA

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
        cs = ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap, **kw)
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.contour(XX, YY, ZZ, levels=levels, colors="k", linewidths=0.4,
               alpha=0.5)

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

    if cmap is None:
        cmap = PARULA

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
        ax.plot_surface(XX, YY, ZZ, cmap=cmap,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True,
                        shade=True, **kw)
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

        ax.pcolormesh(XX, YY, ZZ, cmap=cmap, shading="auto", **kw)
        theta_bdy = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta_bdy), np.sin(theta_bdy), "k-", linewidth=0.8)
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
    **kw,
) -> tuple[plt.Figure, Any]:
    """Plot a Spherefun on the unit sphere (MATLAB Chebfun style).

    Faithful translation of @spherefun/surf.m from MATLAB Chebfun.

    The MATLAB code does::

        l = linspace(-pi, pi, 200);
        t = linspace(0, pi, 200);
        C = fevalm(f, l, t);  % 200x200 grid
        [ll, tt] = meshgrid(l, t);
        [xx,yy,zz] = sph2cart(ll, pi/2 - tt, ones(size(ll)));
        surf(xx, yy, zz, C, 'facecolor','interp', ...);
        daspect([1 1 1]);

    Parameters
    ----------
    fs : Spherefun
    ax : Axes3D, optional
    title : str
    n_pts : int
        Grid resolution (default 200, matching MATLAB ``minPlotNum``).
    cmap : colormap, optional (default: parula)

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp
    from matplotlib.colors import LightSource, Normalize

    if cmap is None:
        cmap = PARULA

    # --- MATLAB: l = linspace(-pi, pi, 200); t = linspace(0, pi, 200) ---
    l = np.linspace(-np.pi, np.pi, n_pts)
    t = np.linspace(0.0, np.pi, n_pts)

    # --- MATLAB: C = fevalm(f, l, t) ---
    # fevalm evaluates on the tensor-product grid.
    # We use pointwise evaluation on the meshgrid.
    ll, tt = np.meshgrid(l, t)  # both (n_pts, n_pts), MATLAB ordering
    C = np.array(
        fs(jnp.array(ll.ravel()), jnp.array(tt.ravel()))
    ).reshape(ll.shape)

    # --- MATLAB: correction for near-constant functions ---
    if np.linalg.norm(C - C[0, 0], ord=np.inf) < 1e-10:
        C = np.full_like(C, C[0, 0])

    # --- MATLAB: [xx,yy,zz] = sph2cart(ll, pi/2 - tt, ones(size(ll))) ---
    # sph2cart(azimuth, elevation, r):
    #   x = r * cos(elevation) * cos(azimuth)
    #   y = r * cos(elevation) * sin(azimuth)
    #   z = r * sin(elevation)
    elev = np.pi / 2 - tt
    xx = np.cos(elev) * np.cos(ll)
    yy = np.cos(elev) * np.sin(ll)
    zz = np.sin(elev)

    # --- MATLAB: surf(xx, yy, zz, C, 'facecolor','interp', ...) ---
    fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36,
                             figsize=(6.1, 5.0))

    # Normalise C for face colours (simulating 'facecolor','interp')
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    fmin, fmax = float(C.min()), float(C.max())
    if fmax > fmin:
        norm_vals = (C - fmin) / (fmax - fmin)
    else:
        norm_vals = np.full_like(C, 0.5)

    # Apply LightSource shading (simulating MATLAB camlight + lighting phong)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = cmap_obj(norm_vals)[:, :, :3]  # drop alpha
    shaded = ls.shade_rgb(rgb, zz)

    # Build RGBA facecolors from shaded RGB
    fcolors = np.ones((*shaded.shape[:2], 4))
    fcolors[:, :, :3] = shaded

    ax.plot_surface(xx, yy, zz, facecolors=fcolors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True,
                    shade=False, **kw)

    # --- MATLAB: daspect([1 1 1]) ---
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


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
    sphere_color: str = "#FFFFCC",
    arrow_color: str = "k",
    arrow_scale: float = 0.15,
    cmap=None,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Quiver plot of a Spherefunv on the unit sphere (MATLAB Chebfun style).

    Draws a light-coloured sphere with black arrows representing the
    tangent vector field.  Arrow length is proportional to the field
    magnitude at each sample point.

    The two Spherefun components are interpreted as:
      - f (component 0): longitudinal (east-west) component
      - g (component 1): latitudinal (north-south) component

    Parameters
    ----------
    fv : Spherefunv
        Vector field on the sphere.
    ax : Axes3D, optional
    title : str
    n_lam, n_theta : int
        Sampling density for the quiver arrows.
    sphere_color : str
        Background sphere surface colour.
    arrow_color : str
        Arrow colour.
    arrow_scale : float
        Scale factor for arrow length.
    cmap : colormap, optional
        If provided, arrows are coloured by magnitude using this colormap
        instead of a single colour.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    # Sample points (avoid poles for cleaner arrows)
    lam = np.linspace(-np.pi, np.pi, n_lam, endpoint=False)
    theta = np.linspace(0.15, np.pi - 0.15, n_theta)
    LAM, THETA = np.meshgrid(lam, theta, indexing="ij")

    lam_flat = jnp.array(LAM.ravel())
    theta_flat = jnp.array(THETA.ravel())

    # Evaluate vector field components
    f_comp, g_comp = fv.components
    f_vals = np.array(f_comp(lam_flat, theta_flat)).reshape(LAM.shape)
    g_vals = np.array(g_comp(lam_flat, theta_flat)).reshape(LAM.shape)

    # Sphere surface points (Cartesian)
    X = np.sin(THETA) * np.cos(LAM)
    Y = np.sin(THETA) * np.sin(LAM)
    Z = np.cos(THETA)

    # Convert tangent vectors (lam, theta) to Cartesian (dx, dy, dz)
    # e_lam = (-sin(lam), cos(lam), 0) / sin(theta)
    # e_theta = (cos(theta)*cos(lam), cos(theta)*sin(lam), -sin(theta))
    sin_th = np.sin(THETA)
    cos_th = np.cos(THETA)
    sin_lam = np.sin(LAM)
    cos_lam = np.cos(LAM)

    # Scale: f_vals is longitudinal, g_vals is latitudinal
    U = f_vals * (-sin_lam) + g_vals * cos_th * cos_lam
    V = f_vals * cos_lam + g_vals * cos_th * sin_lam
    W = -g_vals * sin_th

    # Scale arrows
    mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    max_mag = float(mag.max()) if mag.max() > 0 else 1.0
    U = U * arrow_scale / max_mag
    V = V * arrow_scale / max_mag
    W = W * arrow_scale / max_mag

    # Draw background sphere
    fig, ax = _setup_3d_axes(ax, None, elev=8, azim=-36,
                             figsize=(6.1, 2.58))

    # Light sphere surface
    n_bg = 60
    lam_bg = np.linspace(-np.pi, np.pi, n_bg)
    theta_bg = np.linspace(0, np.pi, n_bg // 2)
    LAM_bg, THETA_bg = np.meshgrid(lam_bg, theta_bg, indexing="ij")
    X_bg = np.sin(THETA_bg) * np.cos(LAM_bg)
    Y_bg = np.sin(THETA_bg) * np.sin(LAM_bg)
    Z_bg = np.cos(THETA_bg)

    ax.plot_surface(X_bg, Y_bg, Z_bg, color=sphere_color,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.3)

    # Quiver arrows
    ax.quiver(X.ravel(), Y.ravel(), Z.ravel(),
              U.ravel(), V.ravel(), W.ravel(),
              color=arrow_color, arrow_length_ratio=0.25,
              linewidth=0.8, **kw)

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
    n_pts: int = 100,
    cmap=None,
    alpha: float = 0.95,
    elev: float = 25,
    azim: float = -37,
    **kw,
) -> tuple[plt.Figure, Any]:
    """Slice plot of a Ballfun inside the unit ball (MATLAB Chebfun style).

    Faithful translation of ``plotBall`` from @ballfun/plot.m.

    The MATLAB code does::

        % Evaluate on grid in spherical coords
        [tt, rr, ll] = meshgrid(th, r, lam);
        % Use slice() to cut at specific r, theta, lambda values
        hslicer = slice(tt, rr, ll, ff, tslice, rslice, lslice);
        % Convert each slice surface from spherical to Cartesian
        for j = 1:numel(hslicer)
            h = hslicer(j);
            [xs,ys,zs] = sph2cart(h.ZData, h.XData, h.YData);
            h = surf(xs, ys, zs, h.CData, 'facecolor','interp', ...);
        end
        camlight('headlight'); lighting phong; material dull;
        axis([-1 1 -1 1 -1 1]); daspect([1 1 1]);

    In MATLAB ``sph2cart(lambda, elevation, r)``, and the slice axes are
    (theta, r, lambda).  The XData/YData/ZData from the slicer carry the
    theta/r/lambda coordinates of each surface patch.

    Since Python has no direct ``slice()`` equivalent, we construct the
    same constant-r, constant-theta, and constant-lambda surfaces
    analytically and convert from spherical to Cartesian.

    Parameters
    ----------
    bf : Ballfun
        The 3-D function on the unit ball.
    ax : Axes3D, optional
    title : str
    n_pts : int
        Grid resolution for each slice surface.
    cmap : colormap, optional (default: parula)
    alpha : float
        Surface transparency.
    elev, azim : float
        Camera view angles.

    Returns
    -------
    fig, ax
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.colors as mcolors
    from matplotlib.colors import LightSource

    if cmap is None:
        cmap = PARULA
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # --- Determine grid sizes (matching MATLAB's size-padding logic) ---
    # MATLAB uses m >= 25, n,p >= 28 and adjusts for divisibility.
    # We simplify to using n_pts for angular resolution and n_pts//2 for radial.
    n_r = max(n_pts // 2, 25)
    n_lam = max(n_pts, 28)
    n_th = max(n_pts, 28)

    # --- Build 1D spherical coordinate arrays ---
    # MATLAB: r = chebpts(m), r = r(floor(m/2)+1:end) -> positive half [0, 1]
    r_full = np.linspace(-1, 1, 2 * n_r)  # simplified; MATLAB uses Chebyshev pts
    r = r_full[n_r:]  # positive half [0+eps, 1]
    r[0] = 0.0  # ensure exact zero

    # MATLAB: lam = [pi*trigpts(n); pi], range around [-pi, pi]
    lam = np.linspace(-np.pi, np.pi, n_lam, endpoint=True)

    # MATLAB: th = th(floor(p/2)+1:end) -> [0, pi]  (colatitude)
    th = np.linspace(0, np.pi, n_th, endpoint=True)

    # --- Slice positions (matching MATLAB logic) ---
    # MATLAB: rslice = r closest to 0.5
    idx_r = np.argmin(np.abs(r - 0.5))
    rslice = [r[idx_r]]
    # MATLAB: tslice = th([1, floor(p/4)+1]) -> theta=0, theta~pi/4
    idx_t2 = max(1, n_th // 4)
    tslice = [th[0], th[idx_t2]]
    # MATLAB: lslice = lam([1, floor(n/4)+1]) -> lam~-pi, lam~-pi/2
    idx_l2 = max(1, n_lam // 4)
    lslice = [lam[0], lam[idx_l2]]

    # --- Helper: evaluate ballfun on a 2D grid of (r, lam, th) ---
    def _eval_grid(bf, r_arr, lam_arr, th_arr):
        """Evaluate bf.fevalm(r_1d, lam_1d, th_1d) and return real values."""
        r_1d = jnp.array(r_arr)
        lam_1d = jnp.array(lam_arr)
        th_1d = jnp.array(th_arr)
        vals = np.asarray(bf.fevalm(r_1d, lam_1d, th_1d))  # (Nr, Nlam, Nth)
        if bf.is_real:
            vals = np.real(vals)
        return vals

    # --- Evaluate on full 3D grid ---
    # MATLAB: ff = real(ballfun.coeffs2vals(F)), permuted to (Nr, Nth, Nlam)
    # fevalm returns (Nr, Nlam, Nth)
    ff_3d = _eval_grid(bf, r, lam, th)  # shape (Nr, Nlam, Nth)

    # --- Build slice surfaces ---
    # Each slice is a 2D surface in spherical coordinates, then converted to Cartesian.
    # MATLAB does sph2cart(lambda, elevation, r) where elevation = pi/2 - theta (colatitude to elevation)

    def _sph2cart(lam_grid, th_grid, r_grid):
        """Convert spherical (lam, theta_colat, r) to Cartesian, matching MATLAB's sph2cart.

        MATLAB sph2cart(az, el, r): x = r*cos(el)*cos(az), y = r*cos(el)*sin(az), z = r*sin(el)
        We have colatitude theta, so elevation = pi/2 - theta.
        """
        el = np.pi / 2 - th_grid
        xs = r_grid * np.cos(el) * np.cos(lam_grid)
        ys = r_grid * np.cos(el) * np.sin(lam_grid)
        zs = r_grid * np.sin(el)
        return xs, ys, zs

    slices = []  # list of (xs, ys, zs, cdata) for each slice surface

    # --- Constant-theta slices (tslice) ---
    # MATLAB: slice along theta dimension -> surface parameterised by (r, lam)
    for th_val in tslice:
        th_idx = np.argmin(np.abs(th - th_val))
        # ff_3d[:, :, th_idx] has shape (Nr, Nlam)
        cdata = ff_3d[:, :, th_idx]  # (Nr, Nlam)
        # Build 2D meshgrid for this slice: axes are r and lam
        R_2d, LAM_2d = np.meshgrid(r, lam, indexing='ij')  # (Nr, Nlam)
        TH_2d = np.full_like(R_2d, th[th_idx])
        xs, ys, zs = _sph2cart(LAM_2d, TH_2d, R_2d)
        slices.append((xs, ys, zs, cdata))

    # --- Constant-r slices (rslice) ---
    # Surface parameterised by (th, lam)
    for r_val in rslice:
        r_idx = np.argmin(np.abs(r - r_val))
        # ff_3d[r_idx, :, :] has shape (Nlam, Nth)
        cdata = ff_3d[r_idx, :, :]  # (Nlam, Nth)
        # Build 2D meshgrid: axes are lam and th
        LAM_2d, TH_2d = np.meshgrid(lam, th, indexing='ij')  # (Nlam, Nth)
        R_2d = np.full_like(LAM_2d, r[r_idx])
        xs, ys, zs = _sph2cart(LAM_2d, TH_2d, R_2d)
        slices.append((xs, ys, zs, cdata))

    # --- Constant-lambda slices (lslice) ---
    # Surface parameterised by (r, th)
    for lam_val in lslice:
        lam_idx = np.argmin(np.abs(lam - lam_val))
        # ff_3d[:, lam_idx, :] has shape (Nr, Nth)
        cdata = ff_3d[:, lam_idx, :]  # (Nr, Nth)
        # Build 2D meshgrid: axes are r and th
        R_2d, TH_2d = np.meshgrid(r, th, indexing='ij')  # (Nr, Nth)
        LAM_2d = np.full_like(R_2d, lam[lam_idx])
        xs, ys, zs = _sph2cart(LAM_2d, TH_2d, R_2d)
        slices.append((xs, ys, zs, cdata))

    # --- Global colour limits across all slices ---
    all_cdata = np.concatenate([s[3].ravel() for s in slices])
    vmin, vmax = float(all_cdata.min()), float(all_cdata.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- MATLAB: camlight('headlight'); lighting phong; material dull ---
    ls = LightSource(azdeg=315, altdeg=45)

    fig, ax = _setup_3d_axes(ax, None, elev=elev, azim=azim,
                             figsize=(6.1, 5.0))

    # --- MATLAB: for j = 1:numel(hslicer); surf(xs,ys,zs,CData,...); end ---
    for xs, ys, zs, cdata in slices:
        rgb = cmap_obj(norm(cdata))[:, :, :3]
        # Use zs as elevation for LightSource shading
        shaded = ls.shade_rgb(rgb, zs)
        fcolors = np.ones((*shaded.shape[:2], 4))
        fcolors[:, :, :3] = shaded
        ax.plot_surface(xs, ys, zs, facecolors=fcolors,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True,
                        alpha=alpha, shade=False)

    # --- MATLAB: axis([-1 1 -1 1 -1 1]); daspect([1 1 1]) ---
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(pad=0.5)
    return fig, ax


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
    from chebfunjax.spherefun.spherefun import Spherefun
    from chebfunjax.spherefun.spherefunv import Spherefunv
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.ballfun.ballfun import Ballfun

    if isinstance(obj, Chebfun):
        return plot_1d(obj, *args, **kwargs)
    elif isinstance(obj, Chebfun2):
        return surf(obj, *args, **kwargs)
    elif isinstance(obj, Spherefun):
        return plot_sphere(obj, *args, **kwargs)
    elif isinstance(obj, Spherefunv):
        return quiver_sphere(obj, *args, **kwargs)
    elif isinstance(obj, Diskfun):
        return plot_disk(obj, *args, **kwargs)
    elif isinstance(obj, Ballfun):
        return obj.plot(*args, **kwargs)
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
