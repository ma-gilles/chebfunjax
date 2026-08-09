"""Low-rank compression of square and round pegs.

Faithful replica of approx2/Pegs.m (Trefethen, 2016): the ranks of
the tilted, square, and round peg functions from cheb.gallery2 --
alignment with the axes decides the rank (tilted high, square 1,
round in between).

Original: https://www.chebfun.org/examples/approx2/Pegs.html
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

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')


def _peg_plot(f, F, k):
    g = np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(g, g)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.contourf(X, Y, f(X, Y), levels=[0, .1, .3, .5, .7, .9, 1.001])
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.text(-.9, .8, f"rank {int(F.rank)}", fontsize=18)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Pegs_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # tilted peg
    def tilted(x, y):
        return 1 / ((1 + (2 * x + .4 * y)**20) * (1 + (2 * y - .4 * x)**20))

    print("fa =")
    print("    @(x,y)1./((1+(2*x+.4*y).^20).*(1+(2*y-.4*x).^20))")
    F = Chebfun2.from_function(tilted)
    print(f"rank {int(F.rank)}")
    _peg_plot(tilted, F, 1)

    # square peg
    def square(x, y):
        return 1 / ((1 + (2 * x)**20) * (1 + (2 * y)**20))

    print("fa =")
    print("    @(x,y)1./((1+(2*x).^20).*(1+(2*y).^20))")
    F = Chebfun2.from_function(square)
    print(f"rank {int(F.rank)}")
    _peg_plot(square, F, 2)

    # round peg
    def roundpeg(x, y):
        return 1 / (1 + ((2 * x)**2 + (2 * y)**2)**10)

    print("fa =")
    print("    @(x,y)1./(1+((2*x).^2+(2*y).^2).^10)")
    F = Chebfun2.from_function(roundpeg)
    print(f"rank {int(F.rank)}")
    _peg_plot(roundpeg, F, 3)


if __name__ == "__main__":
    run()
