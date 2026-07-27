"""Condition number of the Vandermonde quasimatrix.

The continuous Vandermonde "matrix" A = [1, x, x^2, ..., x^n] on [-1, 1] is a
quasimatrix whose columns are monomials.  Its 2-norm condition number (from
the continuous L2 SVD) grows exponentially with n at the rate rho_c^n with
rho_c = 1 + sqrt(2).  Faithful port of Chebfun example linalg/CondVandermonde.m
by Nick Trefethen.

Original: https://www.chebfun.org/examples/linalg/CondVandermonde.html
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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


def vandermonde_quasimatrix(x, n):
    """Quasimatrix A = [x^0, x^1, ..., x^n] on x's domain (MATLAB x.^(0:n))."""
    cols = [x ** k for k in range(n + 1)]
    return Quasimatrix(cols, x.domain)


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/linalg')
    os.makedirs(outdir, exist_ok=True)

    # MATLAB: x = chebfun('x'); A = x.^(0:10); cond(A)
    x = cj.chebfun(lambda x: x)
    A = vandermonde_quasimatrix(x, 10)
    cond_A = A.cond()
    print(f"cond(A) = {cond_A:.15e}")

    # MATLAB: for n = 1:20, c(n) = cond(x.^(0:n)); end
    nn = list(range(1, 21))
    c = [vandermonde_quasimatrix(x, n).cond() for n in nn]

    rhoc = 1.0 + np.sqrt(2.0)
    asymptotics = rhoc ** np.arange(1, 21)

    fig, ax = plt.subplots()
    ax.semilogy(nn, c, color='#0072BD', marker='.', linestyle='-',
                markersize=8, linewidth=1.5, label='Vandermonde matrix')
    ax.semilogy(nn, asymptotics, color='#D95319', marker='.', linestyle='-',
                markersize=8, linewidth=1.5, label='asymptotics')
    ax.grid(True)
    ax.set_xlabel('n')
    ax.set_ylabel('condition number')
    ax.legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'condition_numbers.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # The condition number grows exponentially; asymptotic rate rho_c = 1+sqrt(2)
    assert c[-1] > 1e5, "cond(x.^(0:20)) should be exponentially large"

    print("condition_numbers: done")
    return True


if __name__ == "__main__":
    run()
