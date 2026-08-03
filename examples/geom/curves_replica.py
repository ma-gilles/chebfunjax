"""The minimum distance between two curves.

Faithful replica of geom/Curves.m by Nick Trefethen (April 2017):
two random curves, with the distance between them minimized via a
chebfun2 of |f(x) - g(y)|.

randn draws are not bit-reproducible vs MATLAB; the curves are our
own draws.

Original: https://www.chebfun.org/examples/geom/Curves.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Curves_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    r1 = randnfun(0.5, key=jax.random.PRNGKey(1))
    r2 = randnfun(0.5, key=jax.random.PRNGKey(2))
    ts = np.linspace(-1, 1, 800)
    fv = 1j * ts + 0.2 * np.asarray(r1(ts)) - 1
    gv = 1j * ts + 0.2 * np.asarray(r2(ts)) + 1
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    ax.plot(fv.real, fv.imag, lw=2)
    ax.plot(gv.real, gv.imag, lw=2)
    ax.set_aspect("equal")
    ax.grid(True)
    _save(fig)

    def dist_op(x, y):
        fx = 1j * x + 0.2 * jnp.asarray(r1(x)) - 1
        gy = 1j * y + 0.2 * jnp.asarray(r2(y)) + 1
        return jnp.abs(fx - gy)

    d = cj.chebfun2(dist_op)
    xs = np.linspace(-1, 1, 150)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(d(jnp.asarray(X), jnp.asarray(Y)))
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    cs = ax.contour(X, Y, Z, 20)
    fig.colorbar(cs, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _save(fig)

    mindist, pos = d.min2()
    px, py = float(pos[0]), float(pos[1])
    x0 = 1j * px + 0.2 * float(r1(px)) - 1
    y0 = 1j * py + 0.2 * float(r2(py)) + 1
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    ax.plot(fv.real, fv.imag, lw=2)
    ax.plot(gv.real, gv.imag, lw=2)
    ax.plot([x0.real, y0.real], [x0.imag, y0.imag], '--k', lw=1.2)
    ax.plot([x0.real, y0.real], [x0.imag, y0.imag], '.k', ms=12)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"minimum distance: {float(mindist):g}",
                 fontsize=12)
    _save(fig)
    print(f"minimum distance = {float(mindist):.15f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
