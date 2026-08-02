"""Condition numbers of quasimatrices of polynomial bases.

Faithful replica of linalg/CondNos.m by Nick Trefethen (June 2011):
2-norm condition numbers (continuous SVD) of quasimatrices whose
columns are Chebyshev, Legendre, normalized Legendre, and monomial
bases of degree 0..11.

Original: https://www.chebfun.org/examples/linalg/CondNos.html
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
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.polynomials import chebpoly, legpoly

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

N = 11
FIG = [0]


def _plot_cols(cols):
    FIG[0] += 1
    xs = np.linspace(-1, 1, 600)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for c in cols:
        ax.plot(xs, np.asarray(c(xs)), lw=1.0)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"CondNos_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    cheb_cols = [cj.Chebfun.from_coeffs(jnp.asarray(chebpoly(k)))
                 for k in range(N + 1)]
    _plot_cols(cheb_cols)
    c = Quasimatrix(cheb_cols, cheb_cols[0].domain).cond()
    print(f"Condition no. for Chebyshev polynomials: {c:8.3f}")

    leg_cols = [cj.Chebfun.from_coeffs(jnp.asarray(legpoly(k)))
                for k in range(N + 1)]
    _plot_cols(leg_cols)
    c = Quasimatrix(leg_cols, leg_cols[0].domain).cond()
    print(f"Condition no. for Legendre polynomials: {c:8.3f}")

    legn_cols = [cj.Chebfun.from_coeffs(
        jnp.asarray(legpoly(k, normalize=True)))
        for k in range(N + 1)]
    _plot_cols(legn_cols)
    c = Quasimatrix(legn_cols, legn_cols[0].domain).cond()
    print("Condition no. for normalized Legendre polynomials: "
          f"{c:8.3f}")

    mono_cols = [cj.chebfun(lambda x, _k=k: x**_k)
                 for k in range(N + 1)]
    _plot_cols(mono_cols)
    c = Quasimatrix(mono_cols, mono_cols[0].domain).cond()
    print(f"Condition no. for monomials: {c:8.3f}")


if __name__ == "__main__":
    run()
