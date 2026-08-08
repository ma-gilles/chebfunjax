"""Black-Scholes PDE using operator exponential.

Faithful replica of pde/BSexponential.m by Toby Driscoll (June 2014):
the Black-Scholes equation

    v_t = -(sigma^2/2) s^2 v_ss - r s v_s + r v

on [0, 500] (sigma = 0.45, r = 0.03) with v(0) = 0 and v' -> 1 as
s -> infinity, solved by operator exponential.  The inhomogeneous
right condition is removed by the particular-solution trick
(u = A\\0 with Bu = q; propagate w = v - u homogeneously), and the
payoff v_T = max(0, s - 50) is advanced to t = 0.1..0.5 with expm.

Original: https://www.chebfun.org/examples/pde/BSExponential.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

D = (0.0, 500.0)
SIGMA, R = 0.45, 0.03


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    A = Chebop(lambda s, v: (-SIGMA**2 / 2 * s**2 * v.diff(2)
                             - R * s * v.diff() + R * v), domain=D)
    A.lbc = 0.0
    A.rbc = lambda v: v.diff() - 1     # replaces v -> s as s -> inf

    # Particular solution carrying the inhomogeneous BC.
    u = A.solve(0.0)
    A.rbc = 0.0                        # homogeneous BCs for w = v - u

    s = chebfun(lambda t: t, domain=D)
    vT = (s - 50).maximum(0.0)         # payoff at maturity
    wT = vT - u

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xx = np.linspace(40, 60, 600)
    ax.plot(xx, np.asarray(vT(xx)), lw=2)
    v55 = None
    for t in np.arange(0.1, 0.51, 0.1):
        w = A.expm(-t, wT, n=700)
        v = w + u
        ax.plot(xx, np.asarray(v(xx)), 'k', lw=1.0)
        v55 = float(v(55.0))
    ax.set_ylim(-0.5, 14)
    ax.set_xlim(40, 60)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BSExponential_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Value of the option at s = 55, six months before maturity.
    print("ans =")
    print(f"   {v55:.15f}")


if __name__ == "__main__":
    run()
