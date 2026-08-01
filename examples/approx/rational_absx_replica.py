"""Rational minimax approximation of |x|.

Replica of approx/RationalAbsx.m by Nick Trefethen (March 2017): the
error curve of high-degree rational minimax approximation of |x|,
equioscillating on an exponentially graded scale.

The published example computes type (80,80) in 21.6 s with the
adaptive-barycentric minimax of Filip-Nakatsukasa-Trefethen-Beckermann;
chebfunjax's rational Remez currently converges up to type (30,30) for
this function (ledgered gap), which is what is shown here.

Original: https://www.chebfun.org/examples/approx/RationalAbsx.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

N = 30


def run():
    os.makedirs(_IMG, exist_ok=True)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimax(lambda x: jnp.abs(x), N, rational=True, denom=N,
                    breakpoints=[0.0])
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    print(f"type ({N},{N}) error: {r.err:.6e}")

    lim = 2.5 * r.err
    xx = np.linspace(-1, 1, 3000) ** 3
    ev = np.abs(xx) - np.asarray(r.r(xx))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(xx, ev, lw=2.2)
    ax.grid(True)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"error curve for type ({N},{N}) approximation",
                 fontsize=18)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RationalAbsx_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    xx = np.logspace(-14, 0, 5000)
    ev = np.abs(xx) - np.asarray(r.r(xx))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.semilogx(xx, ev, lw=2.2)
    ax.grid(True)
    ax.axis([1e-14, 1, -lim, lim])
    ax.set_title("semilogx scale", fontsize=18)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RationalAbsx_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
