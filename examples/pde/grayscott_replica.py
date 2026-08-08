"""Gray-Scott equations in 2D.

Faithful replica of pde/GrayScott.m by Nick Trefethen (April 2016):
the coupled reaction-diffusion system

    u_t = ep1 Lap u + b(1-u) - u v^2,
    v_t = ep2 Lap v - d v + u v^2,

on [-1,1]^2 to t = 3500 with spin2/ETDRK4 (N = 200, dt = 2): rolls
("fingerprints") for b = 0.04, d = 0.1; spots for b = 0.025,
d = 0.085; and the published coarser-grid (N = 100) symmetry-breaking
comparison for both.

Original: https://www.chebfun.org/examples/pde/GrayScott.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.spin.solver2d import spin2
from chebfunjax.spin.spinop2 import SpinOp2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

EP1, EP2 = 2e-5, 1e-5
DOM = (-1.0, 1.0, -1.0, 1.0)
FIG = [0]


def _u0(x, y):
    return 1 - np.exp(-80 * ((x + .05)**2 + (y + .02)**2))


def _v0(x, y):
    return np.exp(-80 * ((x - .05)**2 + (y - .02)**2))


def _gs(b, d, N):
    op = SpinOp2(
        lin_coeffs=[(EP1, 0, 0, 0, 0), (EP2, 0, 0, 0, 0)],
        nonlin_vals=[lambda u, v, _b=b: _b * (1 - u) - u * v**2,
                     lambda u, v, _d=d: -_d * v + u * v**2],
        n_vars=2, domain=DOM, tspan=(0.0, 3500.0),
        u0=[_u0, _v0])
    t0 = time.time()
    out = spin2(op, N, 2.0, dealias=False)
    V = out[-1][1]
    print("time_in_seconds =")
    print(f"   {time.time() - t0:.9f}", flush=True)
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.imshow(np.asarray(V).T, origin="lower", cmap="viridis",
              extent=DOM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"GrayScott_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Rolls; 2. Spots (N = 200).
    _gs(0.04, 0.1, 200)
    _gs(0.025, 0.085, 200)

    # 4. Coarser grids break the tilted symmetry.
    _gs(0.04, 0.1, 100)
    _gs(0.025, 0.085, 100)


if __name__ == "__main__":
    run()
