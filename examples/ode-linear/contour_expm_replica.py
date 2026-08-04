"""Exponentials of linear operators via contour integrals.

Faithful replica of ode-linear/ContourExpm.m by Anthony Austin
(May 2013): the heat-equation solution operator exp(t*L) applied to
discontinuous initial data via numerical quadrature of the inverse
Laplace transform along Weideman's optimized Talbot contour — 16
complex-shifted Helmholtz solves per output time.

Original: https://www.chebfun.org/examples/ode-linear/ContourExpm.html
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


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ContourExpm_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    N = 32
    th = np.arange(-N / 2 + 0.5, N / 2) * np.pi / (N / 2)
    a, b, c, d = -0.2407, 0.2387, 0.7409, 0.1349j
    zk = N * (a + b * th / np.tan(c * th) + d * th)
    dzk = (b / np.tan(c * th) - b * c * th / np.sin(c * th)**2 + d)

    # Talbot contour plot
    tt = np.linspace(-np.pi, np.pi, 800)
    gam = N * (a + b * tt / np.tan(c * tt) + d * tt)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot(gam.real, gam.imag, lw=2)
    ax.plot(zk.real, zk.imag, 'rx', ms=8, mew=2)
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title("A Talbot contour", fontsize=14)
    _save(fig)

    # Discontinuous initial data
    u0 = cj.chebfun(
        lambda x: jnp.sign(x - 3 * np.pi / 8)
        * jnp.sign(-(x - 5 * np.pi / 8)) / 2 + 0.5,
        domain=(0, np.pi), splitting=True)
    t = np.linspace(0, np.pi, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u0(t)), 'k', lw=2)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Initial data", fontsize=14)
    ax.grid(True)
    _save(fig)

    # exp(Tf*L) u0 by contour quadrature (symmetry: use upper nodes,
    # take 2*real(../1i) exactly as the published code does)
    Tf = [0.01, 0.1, 0.5, 1.0]
    ufs = []
    for m in range(len(Tf)):
        acc = None
        for k in range(N // 2):
            Ls = Chebop(lambda x, u, _z=complex(zk[k]): _z * u - u.diff(2),
                        domain=(0, np.pi))
            Ls.lbc = 0
            Ls.rbc = 0
            sol = Ls.solve(u0)
            term = sol * complex(np.exp(zk[k] * Tf[m]) * dzk[k])
            acc = term if acc is None else acc + term
        ufs.append((acc * (2 / 1j)).real())
        print(f"Tf = {Tf[m]}: done", flush=True)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u0(t)), 'k', lw=2, label="Initial Data")
    for m, uf in enumerate(ufs):
        ax.plot(t, np.asarray(uf(t)), lw=2, label=f"Tf = {Tf[m]}")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
