"""Pythagorean planets.

Faithful replica of ode-nonlin/ThreePlanets.m by Behnam Hashemi and Nick
Trefethen (December 2014): three bodies released from rest at the
vertices of a 3-4-5 right triangle, attracting each other with pairwise
1/r^2 forces. The orbit is chaotic until t_c ~ 86, when the system
"self-ionizes": one planet leaves in one direction and the other two
depart as a pair in the other. Complex arithmetic is used for brevity,
so each body is a single complex unknown.

Original: https://www.chebfun.org/examples/ode-nonlin/ThreePlanets.html
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

from chebfunjax.operators.chebop import Chebop

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def plotframe(x, y, z, title):
    """Black sky, fixed star field, three coloured planets.

    MATLAB calls rng(0) inside plotframe, so every frame gets the same
    stars. rng(0) seeds the Mersenne twister with 5489, which is
    numpy's RandomState(5489); rand agrees between the two (randn does
    not, but is not used here).
    """
    rs = np.random.RandomState(5489)
    x_stars = 15 * rs.rand(250) - 8
    y_stars = 11 * rs.rand(250) - 3.5

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_facecolor("k")
    ax.fill(20 * np.array([-1, 1, 1, -1, -1]),
            20 * np.array([-1, -1, 1, 1, -1]), "k")
    ax.plot(x_stars, y_stars, ".", color="w", markersize=4)
    ax.plot(np.real(x), np.imag(x), ".", color="r", markersize=35)
    ax.plot(np.real(y), np.imag(y), ".", color="y", markersize=35)
    ax.plot(np.real(z), np.imag(z), ".", color="g", markersize=35)
    ax.axis([1.27 * -6.3, 1.27 * 4.7, -3.5, 7.5])
    ax.set_axis_off()
    ax.set_title(title, fontsize=18)
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"ThreePlanets_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x0, y0, z0 = 0.0, 3.0, 4j
    plotframe(x0, y0, z0, "t = 0")

    def planetfun(t, x, y, z):
        forceYX = (y - x) / abs(y - x) ** 3
        forceZX = (z - x) / abs(z - x) ** 3
        forceZY = (z - y) / abs(z - y) ** 3
        return [x.diff(2) - forceYX - forceZX,
                y.diff(2) + forceYX - forceZY,
                z.diff(2) + forceZX + forceZY]

    tmax = 100
    N = Chebop(planetfun, domain=(0, tmax))
    N.lbc = lambda x, y, z: [x - x0, y - y0, z - z0,
                             x.diff(), y.diff(), z.diff()]
    x, y, z = N.solve(0.0)

    def at(t):
        tt = np.float64(t)
        return (complex(x(tt)), complex(y(tt)), complex(z(tt)))

    for t in (50, 86, tmax):
        xx, yy, zz = at(t)
        plotframe(xx, yy, zz, f"t = {t}")

    # The invariants: equal masses released from rest, so the centre of
    # mass cannot move and the total momentum stays zero.
    for t in (0, 50, 86, tmax):
        xx, yy, zz = at(t)
        print(f"t = {t:3d}:  x = {xx:>19.6f}  y = {yy:>19.6f}  "
              f"z = {zz:>19.6f}")
    c0 = sum(at(0)) / 3.0
    c1 = sum(at(tmax)) / 3.0
    print(f"centre of mass at t=0   : {c0:.12f}")
    print(f"centre of mass at t={tmax}: {c1:.12f}")
    print(f"drift                   : {abs(c1 - c0):.3e}")


if __name__ == "__main__":
    run()
