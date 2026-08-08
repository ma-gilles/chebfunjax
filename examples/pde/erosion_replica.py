"""Heat equation via expm.

Faithful replica of pde/Erosion.m by Nick Trefethen (October 2010):
the heat equation u_t = u_xx on [0, 6] with Neumann conditions,
started from the irregular square-wave sign((-1)^floor(x^1.5)) and
advanced to t = 0.01, 0.02, 0.1 with the operator exponential --
narrower spikes erode faster.

Original: https://www.chebfun.org/examples/pde/Erosion.html
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

D = (0.0, 6.0)
FIG = [0]


def _plot(u, t):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    xx = np.linspace(*D, 4000)
    ax.plot(xx, np.asarray(u(xx)), lw=2)
    ax.set_xlim(*D)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True)
    try:
        ln = sum(len(np.asarray(f.coeffs)) for f in u.funs)
    except Exception:
        ln = -1
    ax.set_title(f"t = {t:4.2f}     length = {ln}", fontsize=16)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Erosion_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    u = chebfun(lambda x: np.sign((-1.0)**np.floor(x**1.5)), domain=D,
                splitting=True)
    _plot(u, 0.0)

    L = Chebop(lambda u: u.diff(2), domain=D)
    L.lbc = lambda u: u.diff()      # Neumann
    L.rbc = lambda u: u.diff()
    dt = 0.01
    u = L.expm(dt, u, n=400)
    _plot(u, 0.01)
    u = L.expm(dt, u, n=400)
    _plot(u, 0.02)
    u = L.expm(8 * dt, u, n=400)
    _plot(u, 0.10)
    xx = np.linspace(*D, 2000)
    print("range at t=0.1:",
          float(np.min(np.asarray(u(xx)))),
          float(np.max(np.asarray(u(xx)))))


if __name__ == "__main__":
    run()
