"""Probability exercises: uniform distributions.

Faithful replica of stats/UniformExercises.m by Jie Gao and Nick
Trefethen (June 2013): quartiles of a uniform density; recovering
(a, b) from mean and variance via chebfun2 roots and via 1D
substitution; and wheel-of-fortune conditional probabilities.

Original: https://www.chebfun.org/examples/stats/UniformExercises.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"UniformExercises_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = cj.chebfun(lambda x: 1.0 + 0 * x, domain=(1.0, 2.0))
    fint = f.cumsum()
    mu_x = float(cj.chebfun(lambda x: x * 1.0,
                            domain=(1.0, 2.0)).sum())
    print("mu_x =")
    print(f"   {mu_x:.15f}")
    a = float(np.asarray((1 - fint - 0.25).roots())[0])
    print("a =")
    print(f"   {a:.15f}")
    print("z =")
    print(f"   {a - mu_x:.15f}")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    xs = np.linspace(a, 2, 100)
    ax.fill_between(xs, np.ones_like(xs), color=(0.3, 0.2, 0.5))
    ax.plot([1, 2], [1, 1], 'k', lw=2)
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 2)
    ax.grid(True)
    _save(fig)

    # recover (a, b) from mean 1 and variance 4/3 via chebfun2 roots
    fmean = cj.chebfun2(lambda a_, b_: (a_ + b_) / 2 - 1.0,
                        domain=(-5, 5, -5, 5))
    gvar = cj.chebfun2(lambda a_, b_: (b_ - a_)**2 / 12 - 4 / 3,
                       domain=(-5, 5, -5, 5))
    r = np.atleast_2d(np.asarray(fmean.roots(gvar)))
    print("r =")
    for row in r:
        print(f"  {row[0]:>19.15f} {row[1]:>19.15f}")
    aa = float(np.min(r[:, 1]))
    bb = float(np.max(r[:, 1]))
    print("a =")
    print(f"  {aa:.15f}")
    print("b =")
    print(f"   {bb:.15f}")
    fu = cj.chebfun(lambda x: 0 * x + 1 / (bb - aa),
                    domain=(aa, bb))
    fintu = fu.cumsum()
    print("p =")
    print(f"   {float(fintu(0.0)):.15f}")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    xs = np.linspace(aa, 0, 100)
    ax.fill_between(xs, np.full_like(xs, 1 / (bb - aa)),
                    color=(0.75, 0.3, 0.2))
    ax.plot([aa, bb], [1 / (bb - aa)] * 2, 'k', lw=1.6)
    ax.set_xlim(aa, bb)
    ax.set_ylim(0, 0.5)
    ax.grid(True)
    _save(fig)

    # 1D approach with b = 2 - a
    g1 = cj.chebfun(lambda a_: ((2 - a_) - a_)**2 / 12 - 4 / 3,
                    domain=(-5.0, 5.0))
    aa_r = np.asarray(g1.roots())
    print("aa =")
    for v in aa_r:
        print(f"  {v:.15f}")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    xs = np.linspace(-5, 5, 400)
    ax.plot(xs, np.asarray(g1(xs)), lw=2)
    ax.plot([-5, 5], [0, 0], '-k')
    ax.plot(aa_r, np.zeros_like(aa_r), 'r.', ms=14)
    ax.grid(True)
    _save(fig)

    # wheel of fortune: uniform on [0, 360]
    f = cj.chebfun(lambda x: 0 * x + 1 / 360, domain=(0.0, 360.0))
    colors = [(0, 5, (1, 0, 0)), (5, 20, (0, 1, 1)),
              (20, 55, (1, 1, 0)), (55, 105, (0, 1, 0)),
              (105, 170, (1, 1, 1)), (170, 250, (0, 0, 1)),
              (250, 360, (0, 0, 0))]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for lo, hi, col in colors:
        xs = np.linspace(lo, hi, 50)
        ax.fill_between(xs, np.full_like(xs, 1 / 360), color=col,
                        edgecolor='gray')
    ax.plot([0, 360], [1 / 360] * 2, 'k', lw=2)
    ax.grid(True)
    _save(fig)
    fint = f.cumsum()
    print("p1 =")
    print(f"   {float(fint(5 + 15)):.15f}")
    print("p1_exact =")
    print(f"   {(5 + 15) / 360:.15f}")

    pnb = 1 - float(fint(80.0))
    print("pnb =")
    print(f"   {pnb:.15f}")
    pnyb = 1 - float(fint(35.0)) - float(fint(110.0))
    print("pnyb =")
    print(f"   {pnyb:.15f}")
    pn = pnyb - float(fint(80.0))
    print("pn =")
    print(f"   {pn:.15f}")
    p2 = pn / pnb
    print("p2 =")
    print(f"   {p2:.15f}")
    print("p2_exact =")
    print(f"   {(1 - (35 + 110 + 80) / 360) / (1 - 80 / 360):.15f}")


if __name__ == "__main__":
    run()
