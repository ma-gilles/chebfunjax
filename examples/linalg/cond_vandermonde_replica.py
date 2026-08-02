"""Conditioning of the Vandermonde quasimatrix.

Faithful replica of linalg/CondVandermonde.m by Nick Trefethen
(June 2019): the condition number of the quasimatrix of monomials
1, x, ..., x^n on [-1,1] grows like (1+sqrt(2))^n.

Original: https://www.chebfun.org/examples/linalg/CondVandermonde.html
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
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def _cond_mono(n):
    cols = [cj.chebfun(lambda x, _k=k: x**_k) for k in range(n + 1)]
    return Quasimatrix(cols, cols[0].domain).cond()


def run():
    os.makedirs(_IMG, exist_ok=True)

    print("ans =")
    print(f"     {_cond_mono(10):.15e}")

    c = np.array([_cond_mono(n) for n in range(1, 21)])
    rhoc = 1 + np.sqrt(2)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ns = np.arange(1, 21)
    ax.semilogy(ns, c, '.-', label="Vandermonde matrix")
    ax.semilogy(ns, rhoc**ns, '.-', label="asymptotics")
    ax.grid(True)
    ax.set_xlabel("n")
    ax.set_ylabel("condition number")
    ax.legend(loc="upper left")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "CondVandermonde_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
