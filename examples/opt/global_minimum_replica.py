"""The SIAM 100-digit challenge global minimum.

Faithful replica of opt/GlobalMinimum.m by Alex Townsend
(March 2013): Problem 4 of the SIAM 100-Digit Challenge — the global
minimum of a violently oscillatory function of two variables — solved
by chebfun2 min2.

Original: https://www.chebfun.org/examples/opt/GlobalMinimum.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')


def f_np(x, y):
    return (np.exp(np.sin(50 * x)) + np.sin(60 * np.exp(y))
            + np.sin(70 * np.sin(x)) + np.sin(np.sin(80 * y))
            - np.sin(10 * (x + y)) + (x**2 + y**2) / 4)


def f_j(x, y):
    return (jnp.exp(jnp.sin(50 * x)) + jnp.sin(60 * jnp.exp(y))
            + jnp.sin(70 * jnp.sin(x)) + jnp.sin(jnp.sin(80 * y))
            - jnp.sin(10 * (x + y)) + (x**2 + y**2) / 4)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = np.linspace(-1, 1, 200)
    xx, yy = np.meshgrid(x, x)
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, yy, f_np(xx, yy), cmap="viridis",
                    rstride=1, cstride=1, linewidth=0)
    ax.set_title("The complicated function", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "GlobalMinimum_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    g = cj.chebfun2(f_j)
    print(f"Rank of function = {g.rank}")

    exact = -3.306868647475237
    s = time.time()
    Y, X = g.min2()
    t = time.time() - s
    print(f"Computed global minimum = {float(Y):1.16f}")
    print(f"Error in Chebfun2 minimum = {abs(float(Y)-exact):1.4e}")
    print(f"Total time taken = {t:1.4f}s")

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    cs = ax.contour(xx, yy, f_np(xx, yy), 30)
    fig.colorbar(cs, ax=ax)
    ax.plot(float(X[0]), float(X[1]), '.k', ms=12)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "GlobalMinimum_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, yy, f_np(xx, yy), cmap="viridis",
                    rstride=2, cstride=2, linewidth=0, alpha=0.9)
    ax.scatter([float(X[0])], [float(X[1])], [float(Y)], c='k',
               s=40)
    ax.set_zlim(-10, 10)
    ax.view_init(elev=4, azim=-24.5)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "GlobalMinimum_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
