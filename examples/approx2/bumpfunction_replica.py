"""The low-rank structure of a sum of bump functions.

Faithful replica of approx2/BumpFunction.m: a sum of 100 randomly
centered Gaussian bumps has numerical rank far below 100 -- the
example's point about low-rank structure in smooth 2D functions.
Sections: growth snapshots at n = 1, 5, 50, 100; the rank; singular
value decay; a cross-section; the y-direction maximum; and the
global max2.

Bump centers use a numpy seed (MATLAB's rng(1) stream is not
reproducible); the rank and decay behavior are the sample-robust
content.

Original: https://www.chebfun.org/examples/approx2/BumpFunction.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')

GAM = 100.0


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(1)
    centers = [(2 * rng.random() - 1, 2 * rng.random() - 1)
               for _ in range(100)]

    def partial_sum(n):
        def g(x, y):
            out = np.zeros(np.shape(x))
            for (x0, y0) in centers[:n]:
                out = out + np.exp(-GAM * ((x - x0)**2 + (y - y0)**2))
            return out
        return g

    # Growth snapshots.
    fig = plt.figure(figsize=(10.4, 8.8))
    gx = np.linspace(-1, 1, 220)
    X, Y = np.meshgrid(gx, gx)
    for j, n in enumerate([1, 5, 50, 100]):
        Z = partial_sum(n)(X, Y)
        ax = fig.add_subplot(2, 2, j + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", rstride=2, cstride=2,
                        linewidth=0)
        ax.set_zlim(0, 5)
        ax.set_title(f"n = {n}")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BumpFunction_repl_01.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    f = Chebfun2.from_function(partial_sum(100))
    print(f"Rank of function is {f.rank}")

    # Singular value decay.
    sv = np.asarray(f.svd() if not isinstance(f.svd(), tuple)
                    else f.svd()[1])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.semilogy(np.arange(1, len(sv) + 1), sv, label="SVD")
    ax.set_title("Decay of singular values of f")
    ax.set_xlabel("Index")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BumpFunction_repl_02.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Cross-section along y = pi/12 and the y-direction maximum.
    g100 = partial_sum(100)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    xs = np.linspace(-1, 1, 1200)
    ax.plot(xs, g100(xs, np.pi / 12 * np.ones_like(xs)), lw=1.4)
    ax.set_title(r"Cross-section along y=$\pi$/12")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BumpFunction_repl_03.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    ys = np.linspace(-1, 1, 700)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = g100(XX, YY)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(xs, ZZ.max(axis=0), lw=1.4)
    ax.set_title("Maximum in the y-direction")
    ax.set_xlabel("x")
    ax.set_ylabel("Maximum")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BumpFunction_repl_04.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    m, X0 = f.max2()
    print(f"max2: {float(m):.6f} at ({float(np.ravel(X0)[0]):.4f}, "
          f"{float(np.ravel(X0)[1]):.4f})", flush=True)


if __name__ == "__main__":
    run()
