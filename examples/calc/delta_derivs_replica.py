"""Delta functions and derivatives.

Faithful replica of calc/DeltaDerivs.m by Nick Trefethen: a sine wave
plus a train of random-magnitude Dirac impulses, its delta-aware
extrema and norms, three successive integrals (staircase, kinks,
smooth), and recovery of the original via a third derivative.

The delta magnitudes are an RNG wall: MATLAB's rng(3) ziggurat randn
stream is not reproducible in NumPy, so norm(f,1) differs in value
(the structure -- Inf extrema, zero mean, Inf 2/inf norms -- is exact).

Original: https://www.chebfun.org/examples/calc/DeltaDerivs.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'calc')


def _plot(f, color, title, stem, deltas=None):
    xs = np.linspace(0, 20, 4000)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), color, lw=1.6)
    if deltas:
        for loc, mag in deltas:
            ax.annotate("", xy=(loc, mag), xytext=(loc, 0),
                        arrowprops=dict(arrowstyle="->", color=color))
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    x = cj.chebfun(lambda t: t, domain=[0, 20])
    f = cj.chebfun(lambda t: 0.5 * jnp.sin(t), domain=[0, 20])
    rng = np.random.RandomState(3)
    for j in range(1, 20):
        f = f + (x - float(j)).dirac() * float(rng.randn())
    f = f - float(f.sum()) / 20.0
    _plot(f, "b", "f:  a sine wave plus a sequence of delta impulses",
          "DeltaDerivs_repl_01", deltas=f.deltas)

    def _p(v):
        print("ans =")
        if np.isposinf(v):
            print("   Inf")
        elif np.isneginf(v):
            print("  -Inf")
        else:
            print(f"    {v:.15e}" if abs(v) < 1e-4 else f"  {v:.15f}")

    _p(f.max()[1])
    _p(f.min()[1])
    _p(float(f.sum()))
    _p(float(f.norm(1)))
    _p(float(f.norm(2)))
    _p(float(f.norm(jnp.inf)))

    g = f.cumsum()
    _plot(g, "r", "The integral of f", "DeltaDerivs_repl_02")
    h = g.cumsum()
    _plot(h, "g", "The second integral of f", "DeltaDerivs_repl_03")
    q = h.cumsum()
    _plot(q, "orange", "The third integral of f", "DeltaDerivs_repl_04")

    f2 = q.diff().diff().diff()
    _plot(f2, "b", "f again, obtained via a third derivative",
          "DeltaDerivs_repl_05", deltas=getattr(f2, "deltas", ()))
    return True


if __name__ == "__main__":
    run()
