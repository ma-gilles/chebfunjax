"""Introducing breakpoints speeds up difficult calculations.

Faithful replica of ode-linear/Breakpoints.m by Nick Trefethen
(November 2016): boundary- and interior-layer BVPs solved first on
plain domains and then with strategically placed breakpoints, showing
dramatic reductions in length; plus a nonlinear shock problem with
zero, one, and two breakpoints.

Original: https://www.chebfun.org/examples/ode-linear/Breakpoints.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
HEAD = "        ep      pos(max(u))    length(u)    time (secs.) "
FS = "%12.1e %14.9f %9d %14.2f"


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Breakpoints_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pos_max(u):
    mx = u.max()
    return float(mx[0]) if isinstance(mx, tuple) else float("nan")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Problem A on plain [0,1], ep = 1e-1..1e-5
    def LA(ep, dom):
        N = Chebop(lambda x, u, _e=ep: -_e * u.diff(2) - u.diff(),
                   domain=dom)
        N.bc = "dirichlet"
        return N

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    print(HEAD, flush=True)
    tt = np.linspace(0, 1, 2000)
    for k in range(1, 6):
        ep = 10.0**(-k)
        t0 = time.time()
        u = LA(ep, (0, 1)).solve(1.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        ax.plot(tt, np.asarray(u(tt)), 'b', lw=1.0)
    ax.grid(True)
    ax.axis([-0.03, 1, 0, 1.03])
    ax.set_title(r"Boundary layers for $\epsilon$ = 1e-1, ..., 1e-5")
    _save(fig)

    # Problem A with a moving breakpoint, ep down to 1e-8
    print(HEAD, flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(1, 9):
        ep = 10.0**(-k)
        dom = (0.0, min(0.5, 40 * ep), 1.0)
        t0 = time.time()
        u = LA(ep, dom).solve(1.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        if k == 3:
            ax.plot(tt, np.asarray(u(tt)), 'b', lw=1.2)
            bp = dom[1]
            ax.plot(bp, float(u(jnp.asarray(bp))), '.r', ms=12)
            u_show = u
    ax.grid(True)
    ax.axis([-0.03, 1, 0, 1.03])
    ax.set_title(r"The same computed with a breakpoint, "
                 r"$\epsilon$ = 1e-3")
    _save(fig)
    print("u =")
    print(repr(u_show))

    # Problem B: interior layers, plain [-2,2]
    def LB(ep, dom):
        N = Chebop(lambda x, u, _e=ep: _e * u.diff(2)
                   + x * u.diff() + x * u, domain=dom)
        N.lbc = -4
        N.rbc = 2
        return N

    t2 = np.linspace(-2, 2, 2400)
    print(HEAD, flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(1, 5):
        ep = 10.0**(-k)
        t0 = time.time()
        u = LB(ep, (-2, 2)).solve(0.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        ax.plot(t2, np.asarray(u(t2)), 'm', lw=1.0)
    ax.grid(True)
    ax.axis([-2, 2, -6, 17])
    ax.set_title(r"Interior layers for $\epsilon$ = 1e-1, ..., 1e-4")
    _save(fig)

    # Problem B with two breakpoints, ep down to 1e-8
    print(HEAD, flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(1, 9):
        ep = 10.0**(-k)
        d = min(0.5, 10 * np.sqrt(ep))
        dom = (-2.0, -d, d, 2.0)
        t0 = time.time()
        u = LB(ep, dom).solve(0.0)
        el = time.time() - t0
        print(FS % (ep, _pos_max(u), len(u), el), flush=True)
        if k == 4:
            ax.plot(t2, np.asarray(u(t2)), 'm', lw=1.2)
            for bp in (-d, d):
                ax.plot(bp, float(u(jnp.asarray(bp))), '.k', ms=12)
            u_show = u
    ax.grid(True)
    ax.axis([-2, 2, -6, 17])
    ax.set_title(r"The same computed with two breakpoints, "
                 r"$\epsilon$ = 1e-4")
    _save(fig)
    print("u =")
    print(repr(u_show))

    # Nonlinear problem with 0, 1, 2 breakpoints
    for doms, title in (
        ((0.0, 1.0), "Nonlinear problem"),
        ((0.0, 1 / 3, 1.0), "Same but with one breakpoint"),
        ((0.0, 0.30, 0.36, 1.0), "Same but with two breakpoints"),
    ):
        N = Chebop(lambda u: 0.005 * u.diff(2) + u * u.diff() - u,
                   domain=doms)
        N.lbc = -7.0 / 6
        N.rbc = 3.0 / 2
        t0 = time.time()
        u = N.solve(0.0)
        el = time.time() - t0
        print("u =")
        print(repr(u))
        print("t =")
        print(f"   {el:.6f}")
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.plot(tt, np.asarray(u(tt)), lw=1.4)
        for bp in doms[1:-1]:
            ax.plot(bp, float(u(jnp.asarray(bp))), '.r', ms=12)
        ax.grid(True)
        ax.set_title(f"{title}: time {el:.2f} secs")
        _save(fig)


if __name__ == "__main__":
    run()
