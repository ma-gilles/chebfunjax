"""Rose curves.

Faithful replica of geom/RoseCurves.m by Grady Wright (June 2015):
rhodonea curves cos(m t / n) e^{it} as periodic chebfuns, the
pi/2 length advantage of trig representations, and a 6x6 garden.

Original: https://www.chebfun.org/examples/geom/RoseCurves.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings
from math import lcm

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"RoseCurves_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def rose(m, n, trig=True):
    L = 2 * np.pi * lcm(m, n)
    return cj.chebfun(
        lambda t: jnp.cos(m / n * t) * jnp.cos(t)
        + 1j * jnp.cos(m / n * t) * jnp.sin(t),
        domain=(0.0, L), trig=trig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    m, n = 50, 51
    f = rose(m, n, trig=True)
    g = rose(m, n, trig=False)
    print("ans =")
    print(f"   {len(g)/len(f):.15f}")
    print("ans =")
    print(f"   {np.pi/2:.15f}")

    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    for mm in range(1, 7):
        for nn in range(1, 7):
            fr = rose(mm, nn)
            L = 2 * np.pi * lcm(mm, nn)
            t = np.linspace(0, L, max(600, 200 * lcm(mm, nn)))
            z = np.asarray(fr(t)) + (2.5 * mm - 2.5j * nn)
            ax.plot(z.real, z.imag, 'k-', lw=0.8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)


if __name__ == "__main__":
    run()
