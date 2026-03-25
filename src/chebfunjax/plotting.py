"""Plotting utilities for chebfunjax.

Provides MATLAB-Chebfun-style plotting functions for Chebfun objects.
All functions return (fig, ax) so callers can overlay additional plots
or customise the figure before saving.

Style constants match the Chebfun blue used by the MATLAB documentation.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

import matplotlib
# Do NOT call matplotlib.use("Agg") unconditionally — that breaks Jupyter
# inline plotting.  Only switch if we are already headless or truly have no
# display available.
if matplotlib.get_backend().lower() == "agg" and not os.environ.get("DISPLAY"):
    pass  # already headless — keep whatever backend is active
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CHEBFUN_BLUE = "#4169E1"   # royal blue — matches MATLAB Chebfun docs
CHEBFUN_RED  = "#E04040"
CHEBFUN_GREEN = "#228B22"
CHEBFUN_ORANGE = "#E08030"

_DEFAULT_LINE_KW: dict[str, Any] = dict(color=CHEBFUN_BLUE, linewidth=1.8)
_DEFAULT_GRID_KW: dict[str, Any] = dict(alpha=0.3, linestyle="--", linewidth=0.6)


def _apply_style(ax: plt.Axes, title: str = "", xlabel: str = "x",
                 ylabel: str = "", grid: bool = True) -> None:
    """Apply clean white-background Chebfun style to an Axes."""
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, **_DEFAULT_GRID_KW)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    f,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "",
    label: str = "",
    color: str = CHEBFUN_BLUE,
    linestyle: str = "-",
    linewidth: float = 1.8,
    n_pts: int = 600,
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 1-D Chebfun on its domain.

    Parameters
    ----------
    f : Chebfun
        The function to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when not provided.
    title, xlabel, ylabel : str
        Axis labels/title.
    label : str
        Line label (for legends).
    color, linestyle, linewidth : plot style.
    n_pts : int
        Number of evaluation points.

    Returns
    -------
    fig, ax
    """
    import jax.numpy as jnp

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    xs = _domain_points(f, n_pts)
    ys = np.array(f(jnp.array(xs)))

    plot_kw: dict[str, Any] = dict(color=color, linewidth=linewidth,
                                   linestyle=linestyle)
    if label:
        plot_kw["label"] = label
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
    **kw,
) -> tuple[plt.Figure, plt.Axes]:
    """Semilogy plot of |Chebyshev coefficients| of *f*.

    Parameters
    ----------
    f : Chebfun
        1-D Chebfun.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str
        Plot title.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    coeffs = np.abs(np.array(f.coeffs))
    ax.semilogy(np.arange(len(coeffs)), coeffs, ".", color=color,
                markersize=4, **kw)

    _apply_style(ax, title=title, xlabel="degree $n$",
                 ylabel="$|a_n|$")
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
