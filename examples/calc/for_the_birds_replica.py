"""Optimizing a bird's flight path.

Faithful replica of calc/ForTheBirds.m by Toby Driscoll: a bird flying
from an island to its nest chooses a landfall point minimizing total
energy; the optimum as a function of the water-to-land energy ratio is
itself a chebfun.

Original: https://www.chebfun.org/examples/calc/ForTheBirds.html
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


def _pm(e, x):
    print("energy_optimal =")
    print(f"   {e:.4f}" if e < 100 else f"  {e:.4f}")
    print("x_optimal =")
    print(f"    {x:.4f}" if x < 10 else f"    {x:.0f}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    water_length = cj.chebfun(lambda x: jnp.sqrt(x ** 2 + 25),
                              domain=[0, 13])
    land_length = cj.chebfun(lambda x: 13 - x, domain=[0, 13])

    total = land_length + water_length * 1.4
    (x_opt, e_opt), _ = total.minandmax()
    _pm(e_opt, x_opt)
    r = np.asarray(total.diff().roots())
    print("ans =")
    print(f"    {float(r[0]):.4f}")

    xs = np.linspace(0, 13, 900)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, np.asarray(total(jnp.asarray(xs))), lw=1.6)
    ax.grid(True)
    ax.set_xlabel("landfall point x")
    ax.set_ylabel("total energy of flight")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ForTheBirds_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    for WL in (1.05, 5.0, 50.0):
        total = land_length + water_length * WL
        (x_opt, e_opt), _ = total.minandmax()
        _pm(e_opt, x_opt)
        if WL == 1.05:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
            a1.plot(xs, np.asarray(total(jnp.asarray(xs))), lw=1.6)
            a1.grid(True)
            a1.set_xlabel("x")
            a1.set_ylabel("total energy")
            d = total.diff()
            a2.plot(xs, np.asarray(d(jnp.asarray(xs))), lw=1.6)
            a2.grid(True)
            a2.set_xlabel("x")
            a2.set_ylabel("derivative")
            fig.set_facecolor("white")
            fig.tight_layout()
            fig.savefig(os.path.join(_IMG, "ForTheBirds_repl_02.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    def optimal_landfall_fn(WL):
        t = land_length + water_length * float(WL)
        rr = np.asarray(t.diff().roots())
        return float(rr[0]) if rr.size else 13.0

    print("ans =")
    print(f"    {optimal_landfall_fn(1.4):.4f}")

    WLs = np.linspace(1.1, 5.0, 160)
    vals = np.array([optimal_landfall_fn(w) for w in WLs])
    optimal_landfall = cj.chebfun(
        jnp.asarray(np.polynomial.chebyshev.chebfit(
            2 * (WLs - 1.1) / 3.9 - 1, vals, 40)), domain=[1.1, 5.0],
        coeffs=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(WLs, vals, lw=1.6)
    ax.grid(True)
    ax.set_xlabel("water-to-land energy ratio")
    ax.set_ylabel("optimal landfall position")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ForTheBirds_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    rts = np.asarray((optimal_landfall - 4.5).roots())
    print("ans =")
    print(f"    {float(rts[0]):.4f}")
    return True


if __name__ == "__main__":
    run()
