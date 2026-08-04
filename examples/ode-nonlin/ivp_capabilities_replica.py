"""IVP capabilities of chebop.

Faithful replica of ode-nonlin/IVPCapabilities.m by Asgeir Birkisson
(May 2016): the van der Pol oscillator solved as an initial-value
problem by time marching, its phase plane and limit cycle, a forced
variant, and the same equation solved instead by collocation.

Original: https://www.chebfun.org/examples/ode-nonlin/IVPCapabilities.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"IVPCapabilities_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    mu = 5.0
    N = Chebop(lambda t, u: u.diff(2) - mu * (1 - u**2) * u.diff() + u,
               domain=(0, 50))
    N.lbc = [0.1, 0]
    t0 = time.time()
    u = N.solve(0.0)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    tt = np.linspace(0, 50, 6000)
    breaks = [float(v) for v in u.domain.breakpoints][1:-1]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(tt, np.asarray(u(tt)), lw=1.2)
    if breaks:
        ax.plot(breaks, np.asarray(u(jnp.asarray(breaks))), 'k.',
                ms=10)
    ax.set_title("Van der Pol oscillator")
    ax.grid(True)
    _save(fig)

    print("u =")
    print(repr(u))

    # Phase plane and limit cycle over the direction field
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.plot(np.asarray(u(tt)), np.asarray(u.diff()(tt)), 'm', lw=1.2)
    N.quiver([-2, 2, -10, 10], ax=ax, n_pts=20)
    ax.set_title("Phase plane and limit cycle")
    ax.grid(True)
    _save(fig)

    # Forced van der Pol
    f = cj.chebfun(lambda t: 5 * jnp.sin(5 * t), domain=(0, 50))
    u_forced = N.solve(f)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.plot(np.asarray(u_forced(tt)),
            np.asarray(u_forced.diff()(tt)), 'm', lw=1.2)
    ax.set_title("Van der Pol oscillator with a nonzero forcing "
                 "function")
    ax.grid(True)
    _save(fig)

    # The same IVP solved by collocation rather than marching
    mu = 1.0
    N2 = Chebop(lambda t, u: u.diff(2) - mu * (1 - u**2) * u.diff() + u,
                domain=(0, 4))
    N2.lbc = [2, 0]
    t0 = time.time()
    u2 = N2.solve(0.0, ivp_solver="chebcolloc2")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    t2 = np.linspace(0, 4, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(t2, np.asarray(u2(t2)), lw=1.2)
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
