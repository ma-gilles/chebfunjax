"""Chebfuns from equispaced data.

Faithful port of approx/EquispacedData.m by Nick Trefethen (April 2015).
Constructs a chebfun from data sampled on an equispaced grid with the ``equi``
flag (a Floater-Hormann / FUNQUI rational interpolant, Gibbs-resistant), and
compares its accuracy against the exact chebfun.

Original: https://www.chebfun.org/examples/approx/EquispacedData.html
Copyright 2015 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the ``equi`` construction now runs (FUNQUI);
the reconstruction is accurate to ~3.5e-6 as published, and the chebfun's
endpoint values (0.31, -2.3) and vertical scale (~2.575) reproduce.  The three
error norms match the published values only to ~4 significant figures
(3.537e-6 vs 3.537098e-6, relerr ~2e-5) and the resolved length differs
(96 vs 99): these are FUNQUI-implementation-dependent details of the
Floater-Hormann blend, not the mathematics -- a documented scheme wall.
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

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    ff = lambda x: jnp.exp(x) * jnp.cos(10 * x) * jnp.tanh(4 * x)
    grid = np.linspace(-1, 1, 40)
    data = np.asarray(ff(jnp.asarray(grid)))

    f = cj.chebfun(data, equi=True)
    fexact = cj.chebfun(ff)

    xx = np.linspace(-1, 1, 20001)
    fex = np.asarray(fexact(xx))

    def _err(g):
        return float(np.max(np.abs(np.asarray(g(xx)) - fex)))

    print("error =")
    print(f"     {_err(f):.15e}")

    # chebfun display: length and endpoint values.
    length = len(np.asarray(f.funs[0].coeffs))
    vscale = float(np.max(np.abs(np.asarray(f(xx)))))
    print("f =")
    print("   chebfun column (1 smooth piece)")
    print("       interval       length     endpoint values  ")
    print(f"[      -1,       1]      {length}      "
          f"{float(f(-1.0)):.2g}     {float(f(1.0)):.2g} ")
    print(f"vscale = {vscale:.6e}.")

    # Reduced degree and loosened tolerance.
    f50 = cj.chebfun(f, n=51)
    print("error50 =")
    print(f"     {_err(f50):.15e}")

    floose = cj.chebfun(data, equi=True, eps=1e-6)
    print("errorloose =")
    print(f"     {_err(floose):.15e}")

    # ------------------------------------------------------------------
    # Plot: the data, the equi chebfun, and its error.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(xx, np.asarray(f(xx)), "b", lw=1.5, label="equi chebfun")
    axes[0].plot(grid, data, ".k", ms=8, label="equispaced data")
    axes[0].set_title("exp(x) cos(10x) tanh(4x), 40 equispaced points",
                      fontsize=9)
    axes[0].legend(fontsize=9)
    axes[1].semilogy(xx, np.abs(np.asarray(f(xx)) - fex) + 1e-18, "r", lw=1.2)
    axes[1].set_title("reconstruction error", fontsize=10)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'EquispacedData.png'), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == '__main__':
    run()
