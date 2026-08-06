"""System of two nonlinear BVPs.

Faithful replica of ode-nonlin/BVPSystem.m by Asgeir Birkisson and Toby
Driscoll (September 2010): a pair of coupled nonlinear ODEs solved two
ways -- with separate unknowns u and v, and with a single indexed
variable.

Original: https://www.chebfun.org/examples/ode-nonlin/BVPSystem.html
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
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"BVPSystem_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _problem():
    N = Chebop(lambda x, u, v: [u.diff(2) - v.sin(), v.diff(2) + u.cos()],
               domain=(-1, 1))
    N.lbc = lambda u, v: [u - 1, v.diff()]
    N.rbc = lambda u, v: [v, u.diff()]
    return N


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- Solution using multiple variables u and v -------------------
    (u, v), info = _problem().solvebvp([0.0, 0.0])
    nrmduvec = info["normDelta"]

    x = np.linspace(-1, 1, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].plot(x, np.asarray(u(x)), lw=2, label="u")
    axes[0].plot(x, np.asarray(v(x)), "--r", lw=2, label="v")
    axes[0].set_title("u and v vs. x", fontsize=10)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlabel("x", fontsize=10)
    axes[0].set_ylabel("u(x) and v(x)", fontsize=10)
    axes[1].semilogy(np.arange(1, len(nrmduvec) + 1), nrmduvec, "-*", lw=2)
    axes[1].set_title("Norm of update vs. iteration no.", fontsize=10)
    axes[1].grid(True)
    axes[1].set_xlabel("iteration no.", fontsize=10)
    axes[1].set_ylabel("norm of update", fontsize=10)
    _save(fig)

    print("normDelta =")
    for d in nrmduvec:
        print(f"  {d:.6e}")
    print(f"u(-1) = {float(u(np.float64(-1.0))):.15f}   (exact 1)")
    print(f"v(1)  = {float(v(np.float64(1.0))):.3e}   (exact 0)")
    print(f"u'(1) = {float(u.diff()(np.float64(1.0))):.3e}   (exact 0)")
    print(f"v'(-1)= {float(v.diff()(np.float64(-1.0))):.3e}   (exact 0)")

    # --- The same problem as one indexed variable --------------------
    # MATLAB writes N.op = @(x,u) [diff(u{1},2) - sin(u{2}); ...], a
    # single chebmatrix unknown. The solution components come back the
    # same way, so indexing the returned pair reproduces u{1}, u{2}.
    sol = _problem().solve([0.0, 0.0])
    u1, u2 = sol[0], sol[1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, np.asarray(u1(x)), lw=2, label="$u_1$")
    ax.plot(x, np.asarray(u2(x)), "--r", lw=2, label="$u_2$")
    ax.set_title("$u_1(x)$ and $u_2(x)$ vs. x", fontsize=10)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("$u_1(x)$ and $u_2(x)$", fontsize=10)
    _save(fig)

    d = max(float(np.max(np.abs(np.asarray(u1(x)) - np.asarray(u(x))))),
            float(np.max(np.abs(np.asarray(u2(x)) - np.asarray(v(x))))))
    print(f"max difference between the two formulations: {d:.3e}")


if __name__ == "__main__":
    run()
