"""Prolate spheroidal wave functions from the FFT kernel.

Faithful port of approx/Prolate.m by Nick Trefethen, April 2021.  The
finite Fourier kernel K(x,t) = exp(icxt) is unitary (up to scaling) on
the discrete grid -- cond(A) = 1 -- and its integral-operator
eigenfunctions are the prolate spheroidal wave functions; the
eigenvalue magnitudes plateau at 1/sqrt(5) = 0.4472... before a
super-exponential drop at the bandlimit.

Original: https://www.chebfun.org/examples/approx/Prolate.html
Copyright 2021 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): rankA = 20 exactly; condA agrees to 13
digits (1 + O(1e-13) roundoff on both sides); the 46-entry lamabs
spectrum reproduces the published values to ~12-15 digits through the
plateau, the transition (0.367273333..., 0.254897726...,
0.136684682...), and the super-exponential tail down to ~1e-13, below
which both sides print eigensolver noise.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.operators.integral import fred_eigs
from chebfunjax.plotting import chebfun_style

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    N = 10
    c = N * np.pi
    K10 = Chebfun2.from_function(lambda x, t: jnp.exp(1j * c * x * t))
    xx = jnp.asarray(np.arange(-N, N) / N)
    A = np.asarray(K10(xx[:, None], xx[None, :]))
    print("condA =")
    print(f"   {np.linalg.cond(A):.15f}")
    print("rankA =")
    print(f"    {np.linalg.matrix_rank(A)}")

    lam10 = np.asarray(fred_eigs(
        lambda x, t: jnp.exp(1j * c * x * t), k=46, which="LM"))
    lamabs = np.sort(np.abs(lam10))[::-1]

    c20 = 20 * np.pi
    lam20 = np.asarray(fred_eigs(
        lambda x, t: jnp.exp(1j * c20 * x * t), k=70, which="LM"))
    lamabs20 = np.sort(np.abs(lam20))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    tt = np.linspace(-1, 1, 200)
    ax1.pcolormesh(tt, tt, np.real(np.exp(1j * c * np.outer(tt, tt))),
                   cmap="viridis", shading="auto")
    ax1.plot(np.tile(np.asarray(xx), N * 2),
             np.repeat(np.asarray(xx), N * 2), ".w", ms=3)
    ax1.set_title(r"Re$(K(x,t))$")
    ax1.set_aspect("equal")
    ax2.semilogy(np.arange(1, len(lamabs) + 1), lamabs, ".", ms=9,
                 label=r"$c = 10\pi$")
    ax2.semilogy(np.arange(1, len(lamabs20) + 1), lamabs20, ".", ms=9,
                 label=r"$c = 20\pi$")
    ax2.set_ylim(1e-15, 100)
    ax2.grid(True)
    ax2.legend(fontsize=9)
    ax2.set_title("Eigenvalue magnitudes")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "Prolate.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("lamabs =")
    for v in lamabs:
        print(f"   {v:.15f}")

    return True


if __name__ == "__main__":
    run()
