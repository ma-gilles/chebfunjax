"""Plotting utilities for chebfunjax — matching MATLAB Chebfun visual style.

Provenance
----------
MATLAB source : @chebfun/plot.m, @chebfun2/plot.m, @chebfun2/surf.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Chebfun default style constants
_CHEBFUN_BLUE = "#4169E1"  # Royal blue — Chebfun's signature color
_LINE_WIDTH = 1.8
_FONT_SIZE = 12
_TITLE_SIZE = 14
_N_PLOT = 2001  # default number of evaluation points (matches Chebfun)
_N_PLOT_2D = 200  # per dimension for 2D


def _apply_chebfun_style(ax):
    """Apply Chebfun-like styling to a matplotlib axes."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=_FONT_SIZE - 2, width=0.8)
    ax.grid(False)


def plot(*args, n=_N_PLOT, title=None, ax=None, **kwargs):
    """Plot one or more Chebfuns.

    Parameters
    ----------
    *args : Chebfun objects, optionally interleaved with format strings
        e.g., ``plot(f)`` or ``plot(f, 'r--', g, 'b-')``
    n : int
        Number of evaluation points.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    _apply_chebfun_style(ax)

    # Parse args: Chebfun objects and optional format strings
    i = 0
    color_idx = 0
    default_colors = [_CHEBFUN_BLUE, "#DC143C", "#228B22", "#FF8C00", "#8B008B"]
    while i < len(args):
        f = args[i]
        fmt = None
        if i + 1 < len(args) and isinstance(args[i + 1], str):
            fmt = args[i + 1]
            i += 2
        else:
            i += 1

        # Get domain and evaluate
        a, b = float(f.domain.a), float(f.domain.b)
        x = jnp.linspace(a, b, n, dtype=jnp.float64)
        y = np.array([float(f(xi)) for xi in x])

        plot_kwargs = {"linewidth": _LINE_WIDTH}
        plot_kwargs.update(kwargs)
        if fmt is None:
            plot_kwargs.setdefault("color", default_colors[color_idx % len(default_colors)])
            color_idx += 1
            ax.plot(np.array(x), y, **plot_kwargs)
        else:
            ax.plot(np.array(x), y, fmt, **plot_kwargs)

    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE)

    fig.tight_layout()
    return fig, ax


def surf(f2, n=_N_PLOT_2D, title=None, ax=None, **kwargs):
    """Surface plot for a Chebfun2.

    Parameters
    ----------
    f2 : Chebfun2 or SeparableApprox
        The 2D function to plot.
    n : int
        Grid resolution per dimension.
    title : str, optional
        Plot title.
    ax : matplotlib Axes3D, optional
    **kwargs
        Passed to ``ax.plot_surface()``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Get domain
    dom = f2.approx.domain if hasattr(f2, "approx") else f2.domain
    xa, xb, ya, yb = dom[0], dom[1], dom[2], dom[3]

    x = np.linspace(xa, xb, n)
    y = np.linspace(ya, yb, n)
    X, Y = np.meshgrid(x, y)

    # Evaluate
    Z = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            Z[i, j] = float(f2(X[i, j], Y[i, j]))

    surf_kwargs = {"cmap": "viridis", "alpha": 0.9, "edgecolor": "none"}
    surf_kwargs.update(kwargs)
    ax.plot_surface(X, Y, Z, **surf_kwargs)

    ax.set_xlabel("x", fontsize=_FONT_SIZE)
    ax.set_ylabel("y", fontsize=_FONT_SIZE)
    ax.tick_params(labelsize=_FONT_SIZE - 2)

    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE)

    fig.tight_layout()
    return fig, ax


def contour(f2, n=_N_PLOT_2D, levels=20, title=None, ax=None, **kwargs):
    """Contour plot for a Chebfun2.

    Parameters
    ----------
    f2 : Chebfun2 or SeparableApprox
    n : int
    levels : int
    title : str, optional
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``ax.contour()``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    _apply_chebfun_style(ax)

    dom = f2.approx.domain if hasattr(f2, "approx") else f2.domain
    xa, xb, ya, yb = dom[0], dom[1], dom[2], dom[3]

    x = np.linspace(xa, xb, n)
    y = np.linspace(ya, yb, n)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            Z[i, j] = float(f2(X[i, j], Y[i, j]))

    contour_kwargs = {"levels": levels, "cmap": "RdBu_r"}
    contour_kwargs.update(kwargs)
    cs = ax.contour(X, Y, Z, **contour_kwargs)
    ax.clabel(cs, inline=True, fontsize=_FONT_SIZE - 3)
    ax.set_xlabel("x", fontsize=_FONT_SIZE)
    ax.set_ylabel("y", fontsize=_FONT_SIZE)
    ax.set_aspect("equal")

    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE)

    fig.tight_layout()
    return fig, ax


def phaseplot(f, n=_N_PLOT_2D, title=None, ax=None, **kwargs):
    """Phase portrait for a complex-valued function.

    Colors pixels by the argument (phase) of f(z) using HSV colormap,
    matching MATLAB Chebfun's phaseplot style.

    Parameters
    ----------
    f : callable
        Complex-valued function f(z) where z = x + iy.
    n : int
        Grid resolution per dimension.
    title : str, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    W = np.array([[complex(f(z)) for z in row] for row in Z])

    # HSV coloring: hue = arg(w), saturation = 1, value based on |w|
    H = (np.angle(W) / (2 * np.pi)) % 1.0  # hue from argument
    S = np.ones_like(H)  # full saturation
    V = 1.0 - 0.5 / (1.0 + np.abs(W))  # brightness from magnitude

    HSV = np.stack([H, S, V], axis=-1)
    RGB = mcolors.hsv_to_rgb(HSV)

    ax.imshow(RGB, extent=[-1, 1, -1, 1], origin="lower", aspect="equal")
    ax.set_xlabel("Re(z)", fontsize=_FONT_SIZE)
    ax.set_ylabel("Im(z)", fontsize=_FONT_SIZE)
    ax.tick_params(labelsize=_FONT_SIZE - 2)

    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE)

    fig.tight_layout()
    return fig, ax
