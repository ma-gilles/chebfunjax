"""Exact Chebyshev coefficients of 1/(5+x).

Faithful replica of cheb/ExactChebCoeffs.m by Mark Richardson (May
2011): the closed-form geometric Chebyshev coefficients of a simple
rational function compared against the computed ones.

Original: https://www.chebfun.org/examples/cheb/ExactChebCoeffs.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def run():
    os.makedirs(_IMG, exist_ok=True)

    fc = cj.chebfun(lambda x: 1.0 / (5 + x))
    n = len(fc)
    k = np.arange(1, n + 1)
    exact = (1 / np.sqrt(6) * (-1.0) ** (k - 1)
             / (5 + np.sqrt(24)) ** (k - 1))
    cheb = np.asarray(fc.coeffs)
    print(f"{'exact':>20} {'chebcoeffs':>20} {'difference':>20}")
    for e, c in zip(exact, cheb):
        print(f"{e:20.15f} {c:20.15f} {e - c:20.15f}")

    c = np.abs(cheb) + 1e-30
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogy(np.arange(len(c)), c, '.-', ms=8, lw=1)
    ax.set_title("Chebyshev coefficients of 1/(5+x)", fontsize=12)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\log(|a_n|)$")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ExactChebCoeffs_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
