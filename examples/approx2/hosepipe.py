"""Combining Chebyshev and trigonometric: source mixed slices and four figures.

Translation of approx2/Hosepipe.m by Nick Trefethen (November2019).
Original: https://www.chebfun.org/examples/approx2/Hosepipe.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, plotcoeffs, surf
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_IMG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "approx2"
)


def _wide_box(ax):
    # Host renderer adapter: MATLAB retains the source rectangular3Daxes
    # position, while Axes3D.apply_aspect forces a square viewport. The
    # supplied plotted coordinates, limits and3Dprojection stay intact.
    def keep_source_position(position=None):
        ax._set_position(
            ax.get_position(original=True) if position is None else position, which="active"
        )

    ax.apply_aspect = keep_source_position


def _figure():
    return plt.figure(figsize=(600 / 72.009, 253 / 72.009))


def _save(fig, index):
    _savefig(fig, os.path.join(_IMG, f"Hosepipe_{index:02d}.png"), size=(600, 253), dpi=72.009)
    plt.close(fig)


def _display(name, obj):
    print(f"{name} =")
    print(obj.disp())


def _plotcoeffs(obj, index):
    # @separableApprox/plotcoeffs.m dispatches actual column slices left,
    # row slices right to1Dplotcoeffs; no coefficient envelope or floor.
    columns, _, rows = obj.cdr()
    fig = _figure()
    for position, slices, title in [
        ([0.13, 0.15, 0.33465909, 0.755], columns, "Column slices"),
        ([0.5703409, 0.15, 0.33465909, 0.755], rows, "Row slices"),
    ]:
        ax = fig.add_axes(position)
        # Historical2019 annulus Fourier slices were rendered as lines;
        # retain explicit per-page override and pinned helper defaults.
        fmt = "-" if index == 4 and title == "Column slices" else "."
        plotcoeffs(
            slices,
            ax=ax,
            source=True,
            title=title,
            fmt=fmt,
            linewidth=0.5,
            markersize=8 if index == 2 and title == "Column slices" else 4,
        )
        # Explicit historical rendering style; the pinned standalone helper
        # marker defaults and all actual coefficient coordinates stay intact.
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.set_position(position)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(labelsize=12)
    _save(fig, index)


def run():
    os.makedirs(_IMG, exist_ok=True)
    r = cj.chebfun(lambda x: 0.5 + 0.04 * jnp.cos(40 * x))
    F = cj.chebfun2(lambda x, phi: 2 * x, trigy=True)
    G = cj.chebfun2(lambda x, phi: r(x) * jnp.cos(jnp.pi * phi), trigy=True)
    H = cj.chebfun2(lambda x, phi: r(x) * jnp.sin(jnp.pi * phi), trigy=True)
    fig = _figure()
    ax = fig.add_subplot(projection="3d")
    surf(F, G, H, ax=ax, n_pts=200)
    # Axis-off tube fills MATLABwideviewport; allworldpoints areinside
    # manualsourceXYZlimits, so retain them past Matplotlibsquareclipbox.
    ax.collections[0].set_clip_on(False)
    ax.collections[0].set_antialiased(False)
    ax.collections[0].set_edgecolor("none")
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.54, 0.54)
    ax.set_zlim(-0.54, 0.54)
    ax.set_box_aspect((4, 1.08, 1.08), zoom=1.65)
    ax.set_axis_off()
    # Source camlight illumination has no exact MATLAB backend equivalent;
    # preserve actual public surface coordinates and color mapping.
    _save(fig, 1)
    _display("F", F)
    _display("G", G)
    _display("H", H)
    _plotcoeffs(G, 2)

    def func(z):
        return (1 + 4 / z**3) ** -1 * (z**3 + 0.1) ** -1

    Fc = cj.chebfun2(
        lambda radius, theta: jnp.abs(func(radius * jnp.exp(1j * theta))),
        domain=(0.5, 1.5, -jnp.pi, jnp.pi),
        trigy=True,
    )
    fig = _figure()
    ax = fig.add_subplot(projection="3d")
    surf(Fc, ax=ax, n_pts=200)
    ax.collections[0].set_antialiased(False)
    ax.collections[0].set_edgecolor("none")
    ax.set_position([0.09, 0.11, 0.69, 0.8])
    _wide_box(ax)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(-float(jnp.pi), float(jnp.pi))
    ax.set_zlim(0, 2)
    ax.set_xticks([0.5, 1, 1.5])
    ax.set_yticks([-2, 0, 2])
    ax.set_zticks([0, 1, 2])
    ax.set_xlabel("r", fontsize=14)
    ax.set_ylabel("t", fontsize=14)
    ax.tick_params(labelsize=12)
    bar = fig.colorbar(ax.collections[0], cax=fig.add_axes([0.84, 0.11, 0.035, 0.815]))
    bar.ax.tick_params(labelsize=12)
    _save(fig, 3)
    _plotcoeffs(Fc, 4)


if __name__ == "__main__":
    run()
