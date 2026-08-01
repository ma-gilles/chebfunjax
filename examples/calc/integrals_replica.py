"""Definite and indefinite integrals.

Faithful replica of calc/Integrals.m by Nick Trefethen: integrals of a
piecewise-constant round(2cos x), partial-interval sums, and the
fundamental theorem of calculus round-trips -- including
cumsum(diff(f)), which recovers f up to the constant f(0) via the
Dirac deltas that diff places at the jumps.

Original: https://www.chebfun.org/examples/calc/Integrals.html
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


def _plot(fs, styles, stem, ylim=None):
    xs = np.linspace(0, 10, 3000)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for f, st in zip(fs, styles):
        ax.plot(xs, np.asarray(f(jnp.asarray(xs))), st, lw=1.2)
    if ylim:
        ax.set_ylim(*ylim)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    f = cj.chebfun(lambda t: 2 * jnp.cos(t), domain=[0, 10]).round()
    _plot([f], ["b"], "Integrals_repl_01", ylim=(-2.5, 2.5))

    print("ans =")
    print(f"  {float(f.sum()):.15f}")
    print("ans =")
    print(f"  {float(f.restrict(3.0, 4.0).sum()):.15f}")

    g = f.cumsum()
    _plot([g], ["m"], "Integrals_repl_02")
    d = (float(np.asarray(g(jnp.asarray([4.0])))[0])
         - float(np.asarray(g(jnp.asarray([3.0])))[0]))
    print("ans =")
    print(f"  {d:.15f}")

    xs = jnp.asarray(np.linspace(0.005, 9.995, 2000))
    r1 = np.asarray((g.diff() - f)(xs))
    print("ans =")
    print(f"     {np.sqrt(np.trapezoid(r1**2, np.asarray(xs))):.0f}")

    h = f.diff().cumsum()
    l2 = float((h - f).norm())
    print("ans =")
    print(f"   {l2:.15f}")
    r2 = np.asarray((h - f)(xs))
    _plot([f, h], ["b", "r"], "Integrals_repl_03", ylim=(-2.5, 2.5))

    f0 = float(np.asarray(f(jnp.asarray([0.0])))[0])
    r3 = np.asarray((h + f0 - f)(xs))
    print("ans =")
    print(f"     {np.sqrt(np.trapezoid(r3**2, np.asarray(xs))):.0f}")
    return True


if __name__ == "__main__":
    run()
