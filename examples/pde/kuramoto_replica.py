"""Kuramoto-Sivashinsky equation and chaos.

Faithful replica of pde/Kuramoto.m by Nick Trefethen (April 2016):

    u_t = -(u^2/2)_x - u_xx - u_xxxx

on [-100, 100] with two Gaussian bumps, solved by spin/ETDRK4
(N = 800; dt = 0.025 symmetric run, 0.05 nonsymmetric run) to
t = 100 and 200 -- provably chaotic dynamics dominated by wavelength
2*sqrt(2)*pi ~ 8.89, with sensitive dependence shown by moving one
bump from x = 50 to 49.9.  Snapshots at t = 100/200 are produced by
chaining two spin runs (restart from the t = 100 grid values), which
reproduces MATLAB's single tspan = [0 100 200] stepping exactly.

Original: https://www.chebfun.org/examples/pde/Kuramoto.html
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
from chebfunjax.spin.solver import spin
from chebfunjax.spin.spinop import SpinOp

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

DOM = (-100.0, 100.0)
N = 800
FIG = [0]


def _ks(u0, dt, t1):
    # spin passes lin_coeff the PHYSICAL angular wavenumbers xi:
    # -u_xx - u_xxxx has symbol xi^2 - xi^4.
    op = SpinOp(lin_coeff=lambda xi: xi**2 - xi**4,
                nonlin_vals=lambda u: -0.5 * u**2,
                nonlin_diff_order=1,
                domain=DOM, tspan=(0.0, t1), u0=u0)
    return spin(op, N, dt, dealias=False)


def _plot(x, u0v, uv, label):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(x, u0v, lw=1.2)
    ax.plot(x, uv, lw=1.2)
    ax.set_xlim(*DOM)
    ax.set_ylim(-4, 4)
    ax.grid(True)
    ax.text(42, 3.4, label, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Kuramoto_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_all = time.time()

    # 1. Symmetric two-bump initial condition, dt = 0.025.
    def u0_sym(x):
        return (np.exp(-((x + 50) / 10)**2)
                + np.exp(-((x - 50) / 10)**2))

    x, _, u100 = _ks(u0_sym, 0.025, 100.0)
    _, _, u200 = _ks(lambda xx: u100, 0.025, 100.0)
    _plot(x, u0_sym(x), u100, "t=0 and t=100")
    _plot(x, u0_sym(x), u200, "t=0 and t=200")
    print(f"symmetric run: sym-error at t=200 = "
          f"{np.max(np.abs(u200 - u200[::-1])):.2e}", flush=True)

    # 2. Symmetry ever so slightly broken, dt = 0.05.
    def u0_ns(x):
        return (np.exp(-((x + 50) / 10)**2)
                + np.exp(-((x - 49.9) / 10)**2))

    t0 = time.time()
    x, _, v100 = _ks(u0_ns, 0.05, 100.0)
    _, _, v200 = _ks(lambda xx: v100, 0.05, 100.0)
    _plot(x, u0_ns(x), v100, "t=0 and t=100")
    _plot(x, u0_ns(x), v200, "t=0 and t=200")

    print("time_elapsed_in_seconds =")
    print(f"  {time.time() - t0:.6f}")
    print(f"(whole example: {time.time() - t_all:.1f}s)")


if __name__ == "__main__":
    run()
