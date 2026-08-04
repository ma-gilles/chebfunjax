"""Lyapunov exponents.

Faithful replica of ode-nonlin/LyapunovExponents.m by Nick Trefethen
(May 2016): two Lorenz trajectories launched 1e-9 apart separate
exponentially, and the growth rate of their distance estimates the
leading Lyapunov exponent of the Lorenz attractor.

Original: https://www.chebfun.org/examples/ode-nonlin/LyapunovExponents.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

DOM = (0.0, 30.0)


def _lorenz(ep):
    N = Chebop(lambda t, x, y, z: [
        x.diff() - 10 * (y - x),
        y.diff() - 28 * x + y + x * z,
        z.diff() + 8 * z / 3 - x * y], domain=DOM)
    N.lbc = lambda x, y, z: [x + 2, y + 3, z - 14 + ep]
    return N.solve(0.0)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    ep = 1e-9
    x1, y1, z1 = _lorenz(0.0)      # 1st trajectory
    x2, y2, z2 = _lorenz(ep)       # 2nd trajectory

    # d = sqrt(|x1-x2|^2 + |y1-y2|^2 + |z1-z2|^2).  For real chebfuns
    # |f|^2 == f^2, and squaring directly avoids the root-finding that
    # abs() would perform on these several-thousand-degree functions.
    d2 = ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    t = np.linspace(DOM[0], DOM[1], 3000)
    dvals = np.sqrt(np.abs(np.asarray(d2(t))))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.semilogy(t, dvals, lw=1.4, label="dist(traj_1, traj_2)")

    # log(d) = log(d^2)/2 on [0, 25], then a degree-1 least-squares fit
    logd = 0.5 * d2.restrict(0, 25).log()
    logd2 = logd.polyfit(1)
    slope = (float(logd2(jnp.array(1.0)))
             - float(logd2(jnp.array(0.0))))
    print("slope =")
    print(f"   {slope:.15f}")

    ax.semilogy(t, 0.8e-9 * np.exp(slope * t), 'k--', lw=1.6,
                label=f"exp({slope:1.2f} x)")
    ax.set_xlabel("time")
    ax.set_title("magnitude of separation of nearby "
                 "Lorenz trajectories")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LyapunovExponents_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
