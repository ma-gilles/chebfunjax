"""Resonant vandalism.

Faithful replica of ode-linear/ResonantVandal.m by Nick Trefethen
(December 2012): a resonantly forced oscillator

    d'' + d = 1 - cos(t),   d(0) = 2, d'(0) = 0,

on [0, 50] — the linearly growing envelope of the resonant response,
the "breakaway time" where the deflection first reaches 20, and the
maximum over [35, 40].

Original: https://www.chebfun.org/examples/ode-linear/ResonantVandal.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    L = Chebop(lambda t, d: d.diff(2) + d - (1 - t.cos()),
               domain=(0, 50))
    L.lbc = lambda d: [d - 2, d.diff()]
    d = L.solve(0.0)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    t = np.linspace(0, 50, 2400)
    ax.plot(t, np.asarray(d(t)), lw=1.6)
    ax.grid(True)
    ax.set_xlabel("t (secs)")
    ax.set_ylabel("d (cm)")
    ax.axis([0, 50, -30, 30])
    ax.plot([0, 50], [20, 20], '--r', lw=2)

    r = np.asarray((d - 20).roots(), dtype=float).ravel()
    breakaway = float(np.min(r))
    print("breakaway_time =")
    print(f"  {breakaway:.15f}")
    ax.plot(breakaway, float(d(jnp.array(breakaway))), '.r', ms=16)
    ax.set_title(f"Breakaway time = {breakaway:.4f}", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ResonantVandal_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    mx = d.restrict(35, 40).max()
    val = mx[1] if isinstance(mx, tuple) else mx
    print("ans =")
    print(f"  {float(val):.15f}")


if __name__ == "__main__":
    run()
