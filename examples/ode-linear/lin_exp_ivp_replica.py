"""A linear exponential initial-value problem.

Faithful replica of ode-linear/LinExpIVP.m (Nick Trefethen and Tom
Maerz, September 2010): u' = lambda u with lambda = -10000 on
[0, 0.005], u(0) = 1, solved with chebop backslash and compared with
exp(lambda x).

Original: https://www.chebfun.org/examples/ode-linear/LinExpIVP.html
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

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    d = (0.0, 0.005)
    lam = -10000.0
    x = cj.chebfun(lambda t: t, domain=d)
    L = Chebop(lambda x, u: u.diff(1) - lam * u, domain=d)
    L.lbc = lambda u: u - 1
    u = L.solve(0.0)
    err = float((u - (lam * x).exp()).norm(jnp.inf))
    print(f"error = {err:7.2e}")

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    t = np.linspace(d[0], d[1], 800)
    ax.plot(t, np.asarray(u(t)), lw=1.6)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("exp(x)", fontsize=12)
    ax.set_title(f"Solution of IVP for exp(x) -- error = {err:7.2e}",
                 fontsize=14)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LinExpIVP_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
