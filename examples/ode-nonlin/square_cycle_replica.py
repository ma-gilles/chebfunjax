"""A square limit cycle.

Faithful replica of ode-nonlin/SquareCycle.m by Nick Trefethen
(June 2016): the planar system

    x' = -(0.2x - y)(x^2 - 1),   y' = -(0.2y + x)(y^2 - 1),

whose trajectories are attracted to the boundary of the unit square.

Original: https://www.chebfun.org/examples/ode-nonlin/SquareCycle.html
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
from chebfunjax.plotting import arrowplot, chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"SquareCycle_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    N = Chebop(lambda t, x, y: [x.diff() + (.2 * x - y) * (x**2 - 1),
                                y.diff() + (.2 * y + x) * (y**2 - 1)],
               domain=(0, 110))
    N.lbc = [.01, .02]
    x, y = N.solve(0.0)

    # Phase plane over the first half of the interval, with an arrowhead
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    arrowplot(x.restrict(0, 50), y.restrict(0, 50), ax=ax)
    ax.axis(1.1 * np.array([-1, 1, -1, 1]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)

    # The two components against time
    t = np.linspace(0, 110, 4000)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(t, np.asarray(x(t)), lw=1.4, label="x")
    ax.plot(t, np.asarray(y(t)), lw=1.4, label="y")
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_xlabel("t")
    ax.legend(loc="lower left")
    ax.grid(True)
    _save(fig)

    # Distance to the unit square, on a log scale
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.semilogy(t, np.abs(1 - np.abs(np.asarray(x(t)))), lw=1.4,
                label="x")
    ax.semilogy(t, np.abs(1 - np.abs(np.asarray(y(t)))), lw=1.4,
                label="y")
    ax.set_ylabel("distance to unit square")
    ax.set_xlabel("t")
    ax.legend(loc="lower left")
    ax.grid(True)
    _save(fig)

    print(f"distance to unit square at t=100: "
          f"{abs(1 - abs(float(x(np.float64(100.0))))):.3e}")


if __name__ == "__main__":
    run()
