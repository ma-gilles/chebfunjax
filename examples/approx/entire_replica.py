"""Chebyshev interpolation of oscillatory entire functions.

Faithful replica of approx/Entire.m by Mark Richardson (September
2010): Bernstein-ellipse estimates predict the chebfun lengths of
sin(N pi x) closely.

Original: https://www.chebfun.org/examples/approx/Entire.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

BLUE = (0.0, 0.45, 0.74)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Bernstein ellipses for rho = 1.1 ... 2.0
    rr = 1 + np.arange(1, 11) / 10
    t = np.linspace(0, 2 * np.pi, 600)
    circ = np.exp(1j * t)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for rho in rr:
        e = (rho * circ + (rho * circ) ** -1) / 2
        ax.plot(e.real, e.imag, color=BLUE, lw=1.1)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Entire_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Degree estimates vs actual chebfun lengths for sin(N pi x)
    ee = float(np.finfo(np.float64).eps)
    NN = np.arange(10, 1011, 100)
    estimates = np.zeros(len(NN))
    chebdegrees = np.zeros(len(NN), dtype=int)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for k, N in enumerate(NN):
        def P(p, N=N):
            return ((jnp.log(2 / ee) - jnp.log(p - 1)
                     + N * jnp.pi / 2 * (p - 1 / p)) / jnp.log(p))
        PP = cj.chebfun(P, domain=(1.01, 10.0))
        pos, mn = PP.min()
        estimates[k] = mn
        ff = cj.chebfun(lambda x, N=N: jnp.sin(jnp.pi * N * x),
                        max_length=2**13)
        chebdegrees[k] = len(ff) - 1
        ps = np.linspace(1.01, 10, 700)
        ax.plot(ps, np.asarray(PP(jnp.asarray(ps))), color=BLUE, lw=1.1)
        ax.plot([pos], [mn], '.r', ms=12)
    for i, ytext in enumerate([200, 800, 1450, 2100, 2700, 3350]):
        ax.text(8.02, ytext, f"N = {NN[i]:3d}")
    ax.set_xlabel(r"$\rho$")
    ax.grid(True)
    ax.set_ylim(0, 3.5e3)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Entire_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    est = np.ceil(estimates).astype(int)
    print("            function        estimate   chebfun length ")
    for k, N in enumerate(NN):
        print(f"         sin( {N:4d} pi x)      {est[k]:4d}"
              f"          {chebdegrees[k]:4d} ")
    print()


if __name__ == "__main__":
    run()
