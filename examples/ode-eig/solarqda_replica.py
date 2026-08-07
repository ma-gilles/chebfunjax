"""Model of a quantum dot array for solar energy.

Faithful replica of ode-eig/SolarQDA.m by Toby Driscoll (May 2011):
the 1D Schroedinger model of a four-well quantum dot array,

    -hbar^2/(2m(x)) psi'' + U(x) psi = E psi,   psi -> 0 far away,

with piecewise-constant potential AND effective mass (InAs/GaAs).  The
four lowest states delocalize across all wells; perturbing the well
depths by 2% (MATLAB rng(1138) values, inlined) localizes them.

Original: https://www.chebfun.org/examples/ode-eig/SolarQDA.html
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

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')

HBAR = 1.054e-34
MM = np.array([0.067, 0.022]) * 9.109e-31 * 1.602e-37
NUMWELL, WIDTH, DEPTH, SPACING = 4, 6.5, 0.85, 3.0

# rng(1138): 0.017*randn(1,4), dumped from MATLAB R2025b.
PERTURB = [-0.014692300539885851, -0.026400680654947582,
           0.015050413362550875, 0.025379229485786824]

x = np.cumsum([0] + [WIDTH, SPACING] * NUMWELL)
x = np.array([-10 * SPACING] + list(x[:-1]) + [x[-1] + 9 * SPACING])


def piecewise_const(vals):
    funs = [_Piece.from_values(jnp.asarray([float(v), float(v)]),
                               float(x[i]), float(x[i + 1]))
            for i, v in enumerate(vals)]
    return Chebfun(funs=funs, domain=Domain(tuple(float(t) for t in x)))


def _solve(Uvals):
    U = piecewise_const(Uvals)
    emass = piecewise_const(list(MM) * NUMWELL + [MM[0]])
    N = Chebop(lambda psi: -HBAR**2 / (2 * emass) * psi.diff(2) + U * psi,
               domain=(x[0], x[-1]))
    N.lbc = 0.0
    N.rbc = 0.0
    lam, Psi = N.eigs(k=NUMWELL, sigma=0, return_eigenfunctions=True)
    lam = np.asarray(lam).real
    idx = np.argsort(lam)
    return U, lam[idx], [Psi[i] for i in idx]


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    Uvals = ([DEPTH, 0.0] * NUMWELL) + [DEPTH]
    U, energies, Psi = _solve(Uvals)

    xx = np.linspace(x[0], x[-1], 3000)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(xx, np.asarray(U(xx)), lw=1.6)
    ax.set_ylabel("potential")
    ax.set_xlim(x[0], x[-1])
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SolarQDA_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("energies =")
    for v in energies:
        print(f"   {v:.15f}")

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.0))
    axes[0].plot(xx, np.asarray(U(xx)), lw=1.6)
    axes[0].set_ylabel("potential")
    axes[0].set_xlim(x[0], x[-1])
    axes[0].grid(True)
    for psi in Psi:
        axes[1].plot(xx, np.asarray(psi(xx)).real, lw=1.6)
    axes[1].set_ylabel("wavefunction")
    axes[1].set_xlim(x[0], x[-1])
    axes[1].grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SolarQDA_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Probability: delocalization over all wells.
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for psi in Psi:
        ax.semilogy(xx, np.asarray(psi(xx)).real**2, lw=1.6)
    ax.set_ylabel("probability")
    ax.set_xlim(x[1], x[-2])
    ax.set_ylim(1e-3, 1e-1)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SolarQDA_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Perturbed well depths: 2% variance localizes the states.
    Uvals = ([DEPTH, 0.0] * NUMWELL) + [DEPTH]
    for i, p in enumerate(PERTURB):
        Uvals[2 * i + 1] += p
    _, energies, Psi = _solve(Uvals)
    print("energies =")
    for v in energies:
        print(f"   {v:.15f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for psi in Psi:
        ax.semilogy(xx, np.asarray(psi(xx)).real**2, lw=1.6)
    ax.set_ylabel("probability")
    ax.set_xlim(x[1], x[-2])
    ax.set_ylim(1e-3, 1e-0)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SolarQDA_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
