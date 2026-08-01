"""A drowning man and Snell's Law.

Faithful replica of calc/SnellsLaw.m by Mohsin Javed: a lifeguard's
optimal entry point into the water minimizes total travel time, and at
the optimum Snell's law of refraction holds.

Original: https://www.chebfun.org/examples/calc/SnellsLaw.html
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


def run():
    os.makedirs(_IMG, exist_ok=True)
    sMan = -5 + 5j
    dMan = 5 - 5j
    vLand, vWater = 10.0, 3.0
    T = cj.chebfun(
        lambda x: jnp.abs(x - sMan) / vLand + jnp.abs(x - dMan) / vWater,
        domain=[np.real(sMan), np.real(dMan)])
    (x0, Tmin), _ = T.minandmax()
    print("Tmin =")
    print(f"   {Tmin:.15f}")
    print("x0 =")
    print(f"   {x0:.15f}")

    xs = np.linspace(-5, 5, 1000)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, np.asarray(T(jnp.asarray(xs))), lw=1.6)
    ax.plot([x0], [Tmin], "or", ms=10)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("Time")
    ax.set_title(f"Optimal Point x_0= {x0:.5f}    "
                 f"Minimum Time = {Tmin:.5f}", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SnellsLaw_repl_01.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.fill_between([-6, 6], [-6, -6], [0, 0], color=(0, 0.8, 1))
    ax.plot([np.real(sMan)], [np.imag(sMan)], ".k", ms=16)
    ax.plot([np.real(dMan)], [np.imag(dMan)], ".r", ms=16)
    ax.plot([np.real(sMan), x0], [np.imag(sMan), 0], "b", lw=1.6)
    ax.plot([x0, np.real(dMan)], [0, np.imag(dMan)], "r", lw=1.6)
    ax.plot([x0, x0], [-4, 4], "--k", lw=1.2)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title("Beach, lifeguard, rescued man", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SnellsLaw_repl_02.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    sinTh1 = abs(np.real(sMan) - x0) / abs(sMan - x0)
    sinTh2 = abs(np.real(dMan) - x0) / abs(dMan - x0)
    print("ans =")
    print(f"    {sinTh1 / vLand - sinTh2 / vWater:.15e}")
    return True


if __name__ == "__main__":
    run()
