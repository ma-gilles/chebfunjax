"""Collective dynamics and consensus.

Faithful replica of ode-random/Consensus.m by Nick Trefethen (May
2017): two particles under smooth random walks starting at +-1,
first independent, then coupled by a short-range attraction
F (u-v) exp(-(u-v)^2) with F = 3 (strong: walks lock together) and
F = 1 (weak: they meet and part).

Sample paths use JAX keys (MATLAB rng(3) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/Consensus.html
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

import jax

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')

DOM = (0.0, 40.0)
FIG = [0]


def _solve_and_plot(op, title):
    FIG[0] += 1
    N = Chebop(op, domain=DOM)
    N.lbc = lambda u, v: [u - 1.0, v + 1.0]
    sol = N.solve(0.0)
    u, v = sol[0], sol[1]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xx = np.linspace(*DOM, 4000)
    ax.plot(xx, np.asarray(u(xx)), lw=2.5)
    ax.plot(xx, np.asarray(v(xx)), lw=2.5)
    ax.grid(True)
    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel("u,v", fontsize=18)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(*DOM)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Consensus_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"fig {FIG[0]} done", flush=True)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    lam = 0.2
    f = randnfun(lam, DOM, big=True, key=jax.random.PRNGKey(30))
    g = randnfun(lam, DOM, big=True, key=jax.random.PRNGKey(31))

    _solve_and_plot(
        lambda t, u, v: [u.diff() + f, v.diff() + g],
        "Two independent random walks")

    for F, ttl in ((3.0, "Walks strongly attracted together"),
                   (1.0, "Walks weakly attracted together")):
        _solve_and_plot(
            lambda t, u, v, _F=F: [
                u.diff() + f + _F * (u - v) * (-(u - v)**2).exp(),
                v.diff() + g + _F * (v - u) * (-(v - u)**2).exp()],
            ttl)

    print("total_time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
