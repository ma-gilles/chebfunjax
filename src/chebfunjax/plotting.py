"""Plotting utilities for chebfunjax.

Provides MATLAB-Chebfun-style plotting functions for Chebfun objects.
All functions return (fig, ax) so callers can overlay additional plots
or customise the figure before saving.

Style constants match the Chebfun blue used by the MATLAB documentation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on HPC login nodes
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
    import jax.numpy as jnp
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Determine domain
    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.array(f2(jnp.array(XX), jnp.array(YY)))

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
    import jax.numpy as jnp

    try:
        x0, x1, y0, y1 = f2.domain
    except Exception:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0

    xs = np.linspace(float(x0), float(x1), n_pts)
    ys = np.linspace(float(y0), float(y1), n_pts)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.array(f2(jnp.array(XX), jnp.array(YY)))

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
