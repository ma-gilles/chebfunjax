"""IVP capabilities of chebop.

Translation of ode-nonlin/IVPCapabilities.m by Asgeir Birkisson
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
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"IVPCapabilities_{FIG[0]:02d}.png"))
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

    breaks = [float(v) for v in u.domain.breakpoints][1:-1]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(u, ax=ax, lw=1.2, color="#0072BD")
    if breaks:
        ax.plot(breaks, np.asarray(u(jnp.asarray(breaks))), 'k.',
                ms=10)
    ax.set_title("Van der Pol oscillator")
    _save(fig)

    print("u =")
    print(repr(u))

    # Phase plane and limit cycle over the direction field
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(u, u.diff(), 'm', ax=ax, lw=1.2)
    N.quiver([-2, 2, -10, 10], ax=ax, n_pts=20)
    ax.set_title("Phase plane and limit cycle")
    _save(fig)

    # Forced van der Pol
    t = cj.chebfun(lambda t: t, domain=(0, 50))
    f = 5 * (5 * t).sin()
    u_forced = N.solve(f)
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(u_forced, u_forced.diff(), 'm', ax=ax, lw=1.2)
    ax.set_title("Van der Pol oscillator with a nonzero forcing "
                 "function")
    _save(fig)

    # The same IVP solved by collocation rather than marching
    mu = 1.0
    N2 = Chebop(lambda t, u: u.diff(2) - mu * (1 - u**2) * u.diff() + u,
                domain=(0, 4))
    N2.lbc = [2, 0]
    t0 = time.time()
    u2 = N2.solve(0.0, ivp_solver="chebcolloc2")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(u2, ax=ax, lw=1.2)
    _save(fig)


if __name__ == "__main__":
    run()
