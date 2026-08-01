"""Best approximation of wiggly functions.

Faithful replica of approx/WigglyApprox.m by Nick Trefethen (February
2013): best approximants of f = T_m + ... + T_n of degree m-1 leave an
equioscillating error concentrated near the endpoints.

Original: https://www.chebfun.org/examples/approx/WigglyApprox.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def fmn(m, n):
    c = np.zeros(n + 1)
    c[m:n + 1] = 1.0
    return cj.chebfun(jnp.asarray(c), coeffs=True)


def _quad(f, p_eval, m, n, zoom, fname):
    t0 = time.time()
    xs = np.linspace(-1, 1, 6000)
    xz = np.linspace(zoom[0], zoom[1], 2500)
    fv = np.asarray(f(jnp.asarray(xs)))
    fz = np.asarray(f(jnp.asarray(xz)))
    res = minimax(lambda x: f(x), m - 1)
    p = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    ev = fv - np.asarray(p(jnp.asarray(xs)))
    ez = fz - np.asarray(p(jnp.asarray(xz)))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    axes[0, 0].plot(xs, fv, lw=1.0)
    axes[0, 0].set_title(f"f({m},{n})", fontsize=14)
    axes[0, 1].plot(xz, fz, lw=1.6)
    axes[0, 1].set_title("closeup", fontsize=14)
    axes[1, 0].plot(xs, ev, 'r', lw=1.0)
    axes[1, 0].set_title("f - p", fontsize=14)
    axes[1, 1].plot(xz, ez, 'r', lw=1.6)
    axes[1, 1].set_title("closeup", fontsize=14)
    for ax in axes.ravel():
        ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    return float(np.max(np.abs(ev)))


def run():
    os.makedirs(_IMG, exist_ok=True)
    e1 = _quad(fmn(30, 40), None, 30, 40, (0.8, 1.0),
               "WigglyApprox_repl_01.png")
    e2 = _quad(fmn(200, 220), None, 200, 220, (0.995, 1.0),
               "WigglyApprox_repl_02.png")
    print(f"err1 = {e1:.6f}, err2 = {e2:.6f}")


if __name__ == "__main__":
    run()
