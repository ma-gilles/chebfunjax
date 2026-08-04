"""Periodic ODE systems.

Faithful replica of ode-linear/PeriodicSystem.m by Hadrien Montanelli
(December 2014): the coupled periodic system

    u - v' = 0,   u'' + v = cos(x),   x in [-pi, pi],

solved with the Fourier discretization, then again on a domain with a
breakpoint (Chebyshev collocation with wrap-around rows), both matching
the exact solution [cos(x+3pi/4), cos(x+pi/4)]/sqrt(2).

Original: https://www.chebfun.org/examples/ode-linear/PeriodicSystem.html
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

FIG = [0]


def _plot(u, v, dom):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.linspace(dom[0], dom[-1], 1200)
    ax.plot(t, np.asarray(u(t)), lw=2, label="u")
    ax.plot(t, np.asarray(v(t)), lw=2, label="v")
    ax.set_title("Solutions u and v", fontsize=14)
    ax.legend()
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"PeriodicSystem_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    def solve(dom):
        A = Chebop(lambda x, u, v: [u - v.diff(), u.diff(2) + v],
                   domain=dom)
        A.bc = "periodic"
        return A.solve([0.0, lambda t: jnp.cos(t)])

    def err(sol, dom):
        x = cj.chebfun(lambda t: t, domain=dom)
        ex1 = (x + 3 * np.pi / 4).cos() / np.sqrt(2)
        ex2 = (x + np.pi / 4).cos() / np.sqrt(2)
        return max(float((sol[0] - ex1).norm(jnp.inf)),
                   float((sol[1] - ex2).norm(jnp.inf)))

    dom = (-np.pi, np.pi)
    sol = solve(dom)
    for s in sol:
        print("ans =")
        print(repr(s))
    _plot(sol[0], sol[1], dom)
    print("err =")
    print(f"     {err(sol, dom):.15e}")

    dom2 = (-np.pi, 0.0, np.pi)
    sol2 = solve(dom2)
    for s in sol2:
        print("ans =")
        print(repr(s))
    _plot(sol2[0], sol2[1], dom2)
    print("err =")
    print(f"     {err(sol2, dom2):.15e}")


if __name__ == "__main__":
    run()
