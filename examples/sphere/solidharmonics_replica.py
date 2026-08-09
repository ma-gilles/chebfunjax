"""Solid harmonics.

Faithful replica of sphere/SolidHarmonics.m: the solid harmonics
R_l^m = r^l Y_l^m are harmonic in the ball (norm(laplacian) at
roundoff), orthonormal under sum3, and cheap to construct even at
degree 150.

Original: https://www.chebfun.org/examples/sphere/SolidHarmonics.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')


def _slice_plot(ax, R, n=140):
    """Slice through y = 0 colored by R (the ballfun plot style)."""
    g = np.linspace(-1, 1, n)
    A, B = np.meshgrid(g, g)
    inside = A**2 + B**2 <= 1
    xi, zi = A[inside], B[inside]
    ri = np.sqrt(xi**2 + zi**2)
    lami = np.where(xi >= 0, 0.0, np.pi)
    thi = np.arccos(np.clip(np.where(ri > 0, zi / np.maximum(ri, 1e-300),
                                     1.0), -1, 1))
    V = np.full(A.shape, np.nan)
    V[inside] = np.asarray(R(ri.reshape(1, -1), lami.reshape(1, -1),
                             thi.reshape(1, -1))).ravel()
    vmax = max(np.nanmax(np.abs(V)), 1e-300)
    ax.imshow(V, origin="lower", cmap="viridis", extent=(-1, 1, -1, 1),
              vmin=-vmax, vmax=vmax)
    ax.set_axis_off()


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    R42 = Ballfun.solharm(4, 2)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    _slice_plot(ax, R42)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SolidHarmonics_repl_01.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    print(f"     {float(R42.laplacian().norm()):.15e}")

    R40 = Ballfun.solharm(4, 0)
    print("ans =")
    print(f"   {float((R42 * R42).sum()):.15f}")
    print("ans =")
    print(f"   {float((R40 * R40).sum()):.15f}")
    print("ans =")
    print(f"     {float((R42 * R40).sum()):.3g}")

    # Table of solid harmonics to degree 3.
    N = 3
    fig, axes = plt.subplots(N + 1, N + 1, figsize=(9.6, 9.2))
    for row in axes:
        for ax in row:
            ax.set_axis_off()
    for l in range(N + 1):
        for m in range(l + 1):
            R = Ballfun.solharm(l, m)
            _slice_plot(axes[l][m], R, n=90)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SolidHarmonics_repl_02.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    t0 = time.time()
    Ballfun.solharm(150, 50)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
