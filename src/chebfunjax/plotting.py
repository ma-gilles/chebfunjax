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
# 1-D plot
# ---------------------------------------------------------------------------

def plot(
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
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "f(x,y)",
    n_pts: int = 80,
    cmap: str = "RdBu_r",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Surface plot of a Chebfun2.

    Parameters
    ----------
    f2 : Chebfun2
        The 2-D function to plot.
    ax : Axes3D, optional
        3-D axes.  A new figure is created when not provided.
    title, xlabel, ylabel, zlabel : str
        Axis labels.
    n_pts : int
        Grid resolution.
    cmap : str
        Colormap name.

    Returns
    -------
    fig, ax
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Determine domain
    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = _eval_2d_vectorized(f2, XX, YY)

    if ax is None:
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    ax.plot_surface(XX, YY, ZZ, cmap=cmap, linewidth=0, antialiased=True,
                    alpha=0.9, **kw)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_zlabel(zlabel, fontsize=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2-D contour plot
# ---------------------------------------------------------------------------

def contour(
    f2,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    n_pts: int = 100,
    levels: int = 12,
    cmap: str = "RdBu_r",
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Filled contour plot of a Chebfun2.

    Returns
    -------
    fig, ax
    """
    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = _eval_2d_vectorized(f2, XX, YY)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.get_figure()

    cs = ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap, **kw)
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.contour(XX, YY, ZZ, levels=levels, colors="k", linewidths=0.4,
               alpha=0.5)

    _apply_style(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=False)
    fig.set_facecolor("white")
    fig.tight_layout()
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
    ax: Optional[plt.Axes] = None,
    title: str = "",
    n_theta: int = 200,
    n_r: int = 100,
    cmap: str = "RdBu_r",
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Pseudocolor plot of a Diskfun in Cartesian coordinates.

    Parameters
    ----------
    fd : Diskfun
        The disk function to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when not provided.
    title : str
        Plot title.
    n_theta : int
        Number of angular grid points.
    n_r : int
        Number of radial grid points.
    cmap : str
        Colormap name.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    r = np.linspace(0.0, 1.0, n_r)
    TT, RR = np.meshgrid(theta, r, indexing="ij")  # (n_theta, n_r)

    # Diskfun.__call__ accepts arrays; evaluate on ravelled grid.
    ZZ = np.array(
        fd(jnp.array(TT.ravel()), jnp.array(RR.ravel()))
    ).reshape(TT.shape)

    # Convert to Cartesian for display.
    XX = RR * np.cos(TT)
    YY = RR * np.sin(TT)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    ax.pcolormesh(XX, YY, ZZ, cmap=cmap, shading="auto", **kw)
    # Draw unit-circle boundary.
    theta_bdy = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta_bdy), np.sin(theta_bdy), "k-", linewidth=0.8)

    ax.set_aspect("equal")
    _apply_style(ax, title=title, xlabel="x", ylabel="y", grid=False)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Sphere function plot (coloured sphere surface)
# ---------------------------------------------------------------------------

def plot_sphere(
    fs,
    ax=None,
    title: str = "",
    n_lam: int = 200,
    n_theta: int = 100,
    cmap: str = "RdBu_r",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Pseudocolor surface plot of a Spherefun on the unit sphere.

    Parameters
    ----------
    fs : Spherefun
        The sphere function to plot.
    ax : Axes3D, optional
        3-D axes.  A new figure is created when not provided.
    title : str
        Plot title.
    n_lam : int
        Number of longitude grid points.
    n_theta : int
        Number of colatitude grid points.
    cmap : str
        Colormap name.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    lam = np.linspace(-np.pi, np.pi, n_lam, endpoint=False)
    theta = np.linspace(0.0, np.pi, n_theta)
    LAM, THETA = np.meshgrid(lam, theta, indexing="ij")  # (n_lam, n_theta)

    ZZ = np.array(
        fs(jnp.array(LAM.ravel()), jnp.array(THETA.ravel()))
    ).reshape(LAM.shape)

    # Spherical -> Cartesian.
    XX = np.sin(THETA) * np.cos(LAM)
    YY = np.sin(THETA) * np.sin(LAM)
    ZZ_cart = np.cos(THETA)

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Normalise function values for face colours.
    fmin, fmax = ZZ.min(), ZZ.max()
    if fmax > fmin:
        norm_vals = (ZZ - fmin) / (fmax - fmin)
    else:
        norm_vals = np.zeros_like(ZZ)

    cmap_obj = plt.get_cmap(cmap)
    fcolors = cmap_obj(norm_vals)

    ax.plot_surface(XX, YY, ZZ_cart, facecolors=fcolors,
                    linewidth=0, antialiased=True, alpha=0.95, **kw)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_zlabel("z", fontsize=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Chebfun3 slice plots
# ---------------------------------------------------------------------------

def plot_slices(
    f3,
    ax=None,
    title: str = "",
    n_pts: int = 80,
    cmap: str = "RdBu_r",
    **kw,
) -> tuple[plt.Figure, Any]:
    """Three orthogonal mid-plane slices of a Chebfun3.

    Plots the z=0, y=0, and x=0 slices (or domain midpoints if the domain
    is not centred on zero) as filled colour images on 3-D axes.

    Parameters
    ----------
    f3 : Chebfun3
        The 3-D function to plot.
    ax : Axes3D, optional
        3-D axes.  A new figure is created when not provided.
    title : str
        Plot title.
    n_pts : int
        Grid resolution for each slice.
    cmap : str
        Colormap name.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

    # Determine global colour limits.
    all_vals = np.concatenate([F_xy.ravel(), F_xz.ravel(), F_yz.ravel()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    import matplotlib.colors as mcolors
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    def _surf_slice(XX, YY, ZZ, F):
        fc = cmap_obj(norm(F))
        ax.plot_surface(XX, YY, ZZ, facecolors=fc, linewidth=0,
                        antialiased=True, alpha=0.85, shade=False)

    # XY slice at z = zm
    _surf_slice(XX_xy, YY_xy, ZM_xy, F_xy)
    # XZ slice at y = ym
    _surf_slice(XX_xz, YM_xz, ZZ_xz, F_xz)
    # YZ slice at x = xm
    _surf_slice(XM_yz, YY_yz, ZZ_yz, F_yz)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_zlabel("z", fontsize=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig, ax


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
