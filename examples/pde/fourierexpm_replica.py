"""Time-dependent PDEs on a periodic interval with expm.

Faithful replica of pde/FourierExpm.m by Hadrien Montanelli (December
2014): u_t = L u solved by the operator exponential u = e^{Lt} u0 --
the convection equation u_t = c(x) u_x with
c = -(1/5 + sin^2(x-1)) to T = 20, and the heat equation u_t = u_xx
to T = 1, both periodic on [0, 2pi], shown as waterfall plots.

Original: https://www.chebfun.org/examples/pde/FourierExpm.html
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

DOM = (0.0, 2 * np.pi)


def _waterfall(us, ts, fname, zlim):
    fig = plt.figure(figsize=(8.6, 5.8))
    ax = fig.add_subplot(projection="3d")
    xx = np.linspace(*DOM, 400)
    for u, tv in zip(us, ts):
        ax.plot(xx, np.full_like(xx, tv), np.asarray(u(xx)),
                color="tab:blue", lw=1.2)
    ax.view_init(70, -80)
    ax.set_xlim(*DOM)
    ax.set_ylim(ts[0], ts[-1])
    ax.set_zlim(*zlim)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Convection: u_t = c(x) u_x, c = -(1/5 + sin^2(x-1)), T = 20.
    ts = np.arange(0, 20.5, 0.5)
    c = chebfun(lambda x: -(1 / 5 + np.sin(x - 1)**2), domain=DOM)
    L = Chebop(lambda x, u: c * u.diff(), domain=DOM)
    L.bc = "periodic"
    u0 = chebfun(lambda x: np.exp(-100 * (x - 1)**2), domain=DOM)
    us = L.expm(ts, u0)
    _waterfall(us, ts, "FourierExpm_repl_01.png", (0, 1))
    print("convection done", flush=True)

    # Heat: u_t = u_xx, T = 1.
    ts = np.arange(0, 1.05, 0.05)
    L = Chebop(lambda u: u.diff(2), domain=DOM)
    L.bc = "periodic"
    u0 = chebfun(lambda x: np.sin(3 * x), domain=DOM)
    us = L.expm(ts, u0)
    _waterfall(us, ts, "FourierExpm_repl_02.png", (-1, 1))

    xx = np.linspace(*DOM, 4000)
    print("ans =")
    print(f"     {np.max(np.abs(np.asarray(us[-1](xx)))):.15e}")


if __name__ == "__main__":
    run()
