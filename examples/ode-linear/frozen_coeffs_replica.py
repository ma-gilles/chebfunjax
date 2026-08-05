"""Frozen coefficients do not determine stability.

Faithful replica of ode-linear/FrozenCoeffs.m by Nick Trefethen
(March 2017): the 2x2 variable-coefficient system

    u' = A(t) u,   A(t) = G(t) B G(t)^{-1} - G'(t)G(t)^{-1}-style
                   rotation of B = [-1 m; 0 -1],  m = 2.2,

whose frozen matrices all have eigenvalue -1 (stable) at every t, yet
whose solutions grow: eigenvalues of the frozen coefficients do not
determine stability of a nonautonomous system.

Original: https://www.chebfun.org/examples/ode-linear/FrozenCoeffs.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import arrowplot, chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    m = 2.2
    L = Chebop(lambda t, u, v: [
        u.diff() - (-1 + m * t.cos() * t.sin()) * u
        - m * t.cos()**2 * v,
        v.diff() - (-m * t.sin()**2) * u
        - (-1 - m * t.cos() * t.sin()) * v],
        domain=(0, 16))
    L.lbc = lambda u, v: [u, v - 1]
    u, v = L.solve(0.0)

    fig, ax = arrowplot(u, v, linewidth=5, markersize=30,
                        ystretch=2)
    ax.grid(True)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "FrozenCoeffs_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
